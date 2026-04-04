#include <chrono>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <cuda.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "tensor.h"
#include "cudnn_helper.h"
#include "rnn_problems.h"

/*
Usage:

The default precision is set based on the architecture and mode.

By default, the program runs the benchmark in training mode.

bin/rnn_bench

To run inference mode, use the following command:

bin/rnn_bench inference


To change the precision for training/inference, use:

bin/rnn_bench train <precision>
bin/rnn_bench inference <precision>

Supported precision types:

For Maxwell GPUS:
float for training and inference

For Pascal GPUS:
float, half for training
float, half, int8 for inference

*/

#ifndef USE_TENSOR_CORES
#if CUDNN_MAJOR >= 7
#define USE_TENSOR_CORES 1
#else
#define USE_TENSOR_CORES 0
#endif
#endif


cudnnHandle_t cudnn_handle;
curandGenerator_t curand_gen;


class cudnnDropout {
    std::shared_ptr<cudnnDropoutDescriptor_t> dropout_desc_;
    std::shared_ptr<Tensor<uint8_t>> dropout_state_;

    struct DropoutDeleter {
        void operator()(cudnnDropoutDescriptor_t * dropout_desc) {
            cudnnDestroyDropoutDescriptor(*dropout_desc);
            delete dropout_desc;
        }
    };

    public:

    cudnnDropout(float dropout_percentage) : dropout_desc_(new cudnnDropoutDescriptor_t,
                                                           DropoutDeleter()) {
        size_t dropoutStateSize;
        CHECK_CUDNN_ERROR(cudnnCreateDropoutDescriptor(dropout_desc_.get()));
        CHECK_CUDNN_ERROR(cudnnDropoutGetStatesSize(cudnn_handle, &dropoutStateSize));

        dropout_state_.reset(new Tensor<uint8_t>(std::vector<int>{static_cast<int>(dropoutStateSize), 1}));

        CHECK_CUDNN_ERROR(cudnnSetDropoutDescriptor(*dropout_desc_,
                                                    cudnn_handle,
                                                    dropout_percentage,
                                                    dropout_state_->begin(),
                                                    dropoutStateSize,
                                                    0ULL) );
    }

    cudnnDropoutDescriptor_t desc() const { return *dropout_desc_; }
};

template <typename T>
class cudnnRNN {
    RNNDescriptor<T> rnn_desc_;
    FilterDescriptorNd<T> wDesc_;
    cudnnDropout dropout_;

    int time_steps_;

    TensorDescriptorNdArray<T> xDescArray_;
    TensorDescriptorNdArray<T> yDescArray_;
    TensorDescriptorNdArray<T> dxDescArray_;
    TensorDescriptorNdArray<T> dyDescArray_;

    TensorDescriptorNd<T> hx_desc_;
    TensorDescriptorNd<T> hy_desc_;
    TensorDescriptorNd<T> dhx_desc_;
    TensorDescriptorNd<T> dhy_desc_;
    TensorDescriptorNd<T> cx_desc_;
    TensorDescriptorNd<T> cy_desc_;
    TensorDescriptorNd<T> dcx_desc_;
    TensorDescriptorNd<T> dcy_desc_;

    size_t weight_size_;
    size_t workspace_size_ = 0;
    size_t train_size_ = 0;

    Tensor<T> weights_;
    Tensor<float> workspace_;
    Tensor<float> trainspace_;
    int m_batch_size;
    public:

    cudnnRNN(int hidden_size, int batch_size, int time_steps, const std::string& rnn_type) :
        dropout_(0.f), time_steps_(time_steps),
        xDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        yDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        dxDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        dyDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        hx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        hy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dhx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dhy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        cx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        cy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dcx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dcy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1})
        {
            m_batch_size = batch_size;

            rnn_desc_ = RNNDescriptor<T>(hidden_size,
                                             1,
                                             dropout_.desc(),
                                             CUDNN_SKIP_INPUT,
                                             CUDNN_UNIDIRECTIONAL,
                                             rnn_type,
                                             cudnn_handle);
            cudnnDataType_t type;
            if (std::is_same<T, float>::value)
                type = CUDNN_DATA_FLOAT;
#if CUDNN_MAJOR >= 6
            else if (std::is_same<T, uint8_t>::value)
                type = CUDNN_DATA_INT8;
#endif
            else if (std::is_same<T, uint16_t>::value)
                type= CUDNN_DATA_HALF;
            else 
                throw std::runtime_error("Unknown type in cudnnRNN constructor.");

            CHECK_CUDNN_ERROR( cudnnGetRNNWeightSpaceSize(cudnn_handle, rnn_desc_.desc(), &weight_size_));

#if (CUDNN_MAJOR >= 8) && (USE_TENSOR_CORES)
            // Get current descriptor parameters first
            cudnnRNNAlgo_t algo;
            cudnnRNNMode_t cellMode;
            cudnnRNNBiasMode_t biasMode;
            cudnnDirectionMode_t dirMode;
            cudnnRNNInputMode_t inputMode;
            cudnnDataType_t dataType;
            cudnnDataType_t mathPrec;
            cudnnMathType_t mathType;
            int inputSize;
            int hiddenSize;
            int projSize;
            int numLayers;
            cudnnDropoutDescriptor_t dropoutDesc;
            uint32_t auxFlags;
            
            // Retrieve existing parameters to preserve them
            CHECK_CUDNN_ERROR( cudnnGetRNNDescriptor_v8(
                rnn_desc_.desc(),
                &algo,
                &cellMode,
                &biasMode,
                &dirMode,
                &inputMode,
                &dataType,
                &mathPrec, 
                &mathType,
                &inputSize,
                &hiddenSize,
                &projSize,
                &numLayers,
                &dropoutDesc,
                &auxFlags
            ));
            
            // Set descriptor with tensor cores enabled
            CHECK_CUDNN_ERROR( cudnnSetRNNDescriptor_v8(
                rnn_desc_.desc(),
                algo,
                cellMode,
                biasMode,
                dirMode,
                inputMode,
                dataType,
                mathPrec,
                CUDNN_TENSOR_OP_MATH,
                inputSize,
                hiddenSize,
                projSize,
                numLayers,
                dropoutDesc,
                auxFlags
            ));
#elif (CUDNN_MAJOR >= 8)
    // Get current descriptor parameters first
    cudnnRNNAlgo_t algo;
    cudnnRNNMode_t cellMode;
    cudnnRNNBiasMode_t biasMode;
    cudnnDirectionMode_t dirMode;
    cudnnRNNInputMode_t inputMode;
    cudnnDataType_t dataType;
    cudnnDataType_t mathPrec;
    cudnnMathType_t mathType;
    int inputSize;
    int hiddenSize;
    int projSize;
    int numLayers;
    cudnnDropoutDescriptor_t dropoutDesc;
    uint32_t auxFlags;
    
    // Retrieve existing parameters to preserve them
    CHECK_CUDNN_ERROR( cudnnGetRNNDescriptor_v8(
        rnn_desc_.desc(),
        &algo,
        &cellMode,
        &biasMode,
        &dirMode,
        &inputMode,
        &dataType,
        &mathPrec, 
        &mathType,
        &inputSize,
        &hiddenSize,
        &projSize,
        &numLayers,
        &dropoutDesc,
        &auxFlags
    ));
    
    // Set descriptor with tensor cores enabled
    CHECK_CUDNN_ERROR( cudnnSetRNNDescriptor_v8(
        rnn_desc_.desc(),
        algo,
        cellMode,
        biasMode,
        dirMode,
        inputMode,
        dataType,
        mathPrec,
        CUDNN_DEFAULT_MATH,
        inputSize,
        hiddenSize,
        projSize,
        numLayers,
        dropoutDesc,
        auxFlags
    ));
#elif (CUDNN_MAJOR >= 7) && (USE_TENSOR_CORES)
    // Fall back to deprecated API for older cuDNN versions
    CHECK_CUDNN_ERROR( cudnnSetRNNMatrixMathType(rnn_desc_.desc(), CUDNN_TENSOR_OP_MATH) );
#endif

            weights_ = rand<T>(std::vector<int>{static_cast<int>(weight_size_ / sizeof(T)), 1}, curand_gen);
            
            // --- Calculate workspace and trainspace sizes --- //
            cudnnRNNDataDescriptor_t rnnDataDesc;
            CHECK_CUDNN_ERROR(cudnnCreateRNNDataDescriptor(&rnnDataDesc));

            // Determine correct data type based on T
            cudnnDataType_t dataTypeCompute;
            if (std::is_same<T, float>::value)
                dataTypeCompute = CUDNN_DATA_FLOAT;
#if CUDNN_MAJOR >= 6
            else if (std::is_same<T, uint8_t>::value)
                dataTypeCompute = CUDNN_DATA_INT8;
#endif
            else if (std::is_same<T, uint16_t>::value)
                dataTypeCompute = CUDNN_DATA_HALF;
            else
                throw std::runtime_error("Unknown type T for rnnDataDesc setup.");

            // Get input vector size from xDescArray_ (assuming it's already initialized)
            int n, c, h, w, nStride, cStride, hStride, wStride;
            int inputVecSize; // Use this for vectorSize parameter
            cudnnGetTensor4dDescriptor(xDescArray_.ptr()[0], &dataTypeCompute, &n, &c, &h, &w,
                                       &nStride, &cStride, &hStride, &wStride);
            inputVecSize = c; // Input vector size is the feature dimension (channel)

            // Correctly setup seqLengthArray for batch
            std::vector<int> hostSeqLengths(batch_size);
            for (int i = 0; i < batch_size; i++) {
                hostSeqLengths[i] = time_steps_;
            }

            float paddingFill = 0.0f;

            // Set up the descriptor with appropriate values
            CHECK_CUDNN_ERROR(cudnnSetRNNDataDescriptor(
                rnnDataDesc,
                dataTypeCompute,                   // Use derived data type
                CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, // layout
                time_steps_,                       // maxSeqLength
                batch_size,                        // batchSize
                inputVecSize,                      // Use derived input vector size
                hostSeqLengths.data(),             // Use host sequence lengths array
                &paddingFill                       // paddingFill
            ));

            std::vector<int> dim = {(int)(weight_size_ / sizeof(T)), 1, 1}; // Cast size_t to int
            wDesc_ = FilterDescriptorNd<T>(CUDNN_TENSOR_NCHW, dim);

            // Get workspace and trainspace (reserve space) sizes using the correct descriptor
            CHECK_CUDNN_ERROR(cudnnGetRNNTempSpaceSizes(cudnn_handle,
                rnn_desc_.desc(),
                CUDNN_FWD_MODE_TRAINING, // Assuming training mode for size calculation
                rnnDataDesc,
                &workspace_size_,
                &train_size_
            ));

            // Allocate workspace and trainspace (reserve space)
            // Note: cuDNN docs typically specify workspace as float, reserve space might vary
            // Assuming reserve space is also float based on original code, adjust if needed
            workspace_ = zeros<float>(std::vector<int>{static_cast<int>(workspace_size_ / sizeof(float)), 1});
            trainspace_ = zeros<float>(std::vector<int>{static_cast<int>(train_size_ / sizeof(float)), 1});

            // Clean up the temporary descriptor
            CHECK_CUDNN_ERROR(cudnnDestroyRNNDataDescriptor(rnnDataDesc));
        }
        void forward(Tensor<T> x, Tensor<T> hx, Tensor<T> cx,
                     Tensor<T> y, Tensor<T> hy, Tensor<T> cy) {
            int* hostSeqLengths = new int[m_batch_size];
            for (int i = 0; i < m_batch_size; i++) {
                hostSeqLengths[i] = time_steps_; // Set the length for sequence i
            }
            
            // Allocate and copy to device
            int* devSeqLengths;
            cudaMalloc(&devSeqLengths, m_batch_size * sizeof(int));
            cudaMemcpy(devSeqLengths, hostSeqLengths, m_batch_size * sizeof(int), cudaMemcpyHostToDevice);
            cudnnRNNDataDescriptor_t xDataDesc;
            cudnnRNNDataDescriptor_t yDataDesc;
            CHECK_CUDNN_ERROR(cudnnCreateRNNDataDescriptor(&xDataDesc));
            CHECK_CUDNN_ERROR(cudnnCreateRNNDataDescriptor(&yDataDesc));
            cudnnDataType_t dataType = CUDNN_DATA_HALF;
            // Configure input data descriptor
            float paddingFill = 0.0f;
            
            // Extract dimensions from tensor descriptors
            // Note: xDescArray_ contains one descriptor per time step, all with the same dimensions
            int inputSize = 0;
            int n, c, h, w, nStride, cStride, hStride, wStride;
            
            // Get dimensions from the first descriptor in the array
            // For cuDNN v7, the tensor descriptor dimensions should be [batch_size, inputSize, 1]
            cudnnGetTensor4dDescriptor(xDescArray_.ptr()[0], &dataType, &n, &c, &h, &w, 
                                    &nStride, &cStride, &hStride, &wStride);
            
            inputSize = c; // Input size is the number of features per time step
            
            // Set the RNN data descriptors
            CHECK_CUDNN_ERROR(cudnnSetRNNDataDescriptor(
                xDataDesc,
                dataType,
                CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                time_steps_,
                m_batch_size,
                inputSize,
                hostSeqLengths,
                &paddingFill
            ));
            
            // Output descriptor has same dimensions but possibly different data type
            CHECK_CUDNN_ERROR(cudnnSetRNNDataDescriptor(
                yDataDesc,
                dataType,
                CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                time_steps_,
                m_batch_size,
                inputSize,  // Should match the hidden size for your RNN
                hostSeqLengths,
                &paddingFill
            ));

            CHECK_CUDNN_ERROR( cudnnRNNForward(cudnn_handle,
                                                       rnn_desc_.desc(),
                                                       CUDNN_FWD_MODE_TRAINING,
                                                       devSeqLengths,
                                                       xDataDesc,
                                                       (void *)x.begin(),
                                                       yDataDesc,
                                                       (void *)y.begin(),
                                                       hx_desc_.desc(),
                                                       (void *)hx.begin(),
                                                       (void *)hy.begin(),
                                                       cy_desc_.desc(),
                                                       (void *)cx.begin(),
                                                       (void *)cy.begin(),
                                                       weight_size_,
                                                       (void *)weights_.begin(),
                                                       workspace_size_,
                                                       (void *)workspace_.begin(),
                                                       train_size_,
                                                       (void *)trainspace_.begin()
                                                       ) );
            // Clean up descriptors
            CHECK_CUDNN_ERROR(cudnnDestroyRNNDataDescriptor(yDataDesc));
            CHECK_CUDNN_ERROR(cudnnDestroyRNNDataDescriptor(xDataDesc));
            delete[] hostSeqLengths;
            cudaFree(devSeqLengths);
        }
        void backward_data(Tensor<T> y, Tensor<T> dy, Tensor<T> dhy,
                           Tensor<T> dcy, Tensor<T> hx, Tensor<T> cx,
                           Tensor<T> dx, Tensor<T> dhx, Tensor<T> dcx) {
            cudnnRNNDataDescriptor_t yDataDesc, xDataDesc;
            CHECK_CUDNN_ERROR(cudnnCreateRNNDataDescriptor(&yDataDesc));
            CHECK_CUDNN_ERROR(cudnnCreateRNNDataDescriptor(&xDataDesc));
            int* hostSeqLengths = new int[m_batch_size];
            for (int i = 0; i < m_batch_size; i++) {
                hostSeqLengths[i] = time_steps_; // Set the length for sequence i
            }
            
            // Allocate and copy to device
            int* devSeqLengths;
            cudaMalloc(&devSeqLengths, m_batch_size * sizeof(int));
            cudaMemcpy(devSeqLengths, hostSeqLengths, m_batch_size * sizeof(int), cudaMemcpyHostToDevice);
            // Configure RNN data descriptors
            // Get dataType and vectorSize from existing descriptors (like in forward pass)
            cudnnDataType_t dataType;
            int n, c, h, w, nStride, cStride, hStride, wStride;
            int vectorSize; // Use this for vectorSize parameter
            // Using dxDescArray_ assuming it has the correct input dimensions/type.
            // Need to check if dyDescArray_ should be used for yDataDesc if output size != input size
            cudnnGetTensor4dDescriptor(dxDescArray_.ptr()[0], &dataType, &n, &c, &h, &w,
                                       &nStride, &cStride, &hStride, &wStride);
            vectorSize = c; // Vector size is the feature dimension (channel)
            float paddingFill = 0.0f; // Define paddingFill

            // For y descriptor (output gradients dy)
            CHECK_CUDNN_ERROR(cudnnSetRNNDataDescriptor(
                yDataDesc,
                dataType,                           // Use derived dataType
                CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, // Match forward pass layout
                time_steps_,                        // maxSeqLength
                m_batch_size,                       // Use member batch size
                vectorSize,                         // Use derived vector size
                hostSeqLengths,                     // Pass sequence lengths array
                &paddingFill                        // Pass padding fill pointer
            ));

            // For x descriptor (input gradients dx)
            CHECK_CUDNN_ERROR(cudnnSetRNNDataDescriptor(
                xDataDesc,
                dataType,                           // Use derived dataType
                CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, // Match forward pass layout
                time_steps_,                        // maxSeqLength
                m_batch_size,                       // Use member batch size
                vectorSize,                         // Use derived vector size
                hostSeqLengths,                     // Pass sequence lengths array
                &paddingFill                        // Pass padding fill pointer
            ));

            // Call the v8 version of the RNN backward data
            CHECK_CUDNN_ERROR(cudnnRNNBackwardData_v8(
                cudnn_handle,
                rnn_desc_.desc(),
                devSeqLengths,                    // device array of sequence lengths
                yDataDesc,                        // descriptor for y data
                (void *)y.begin(),                // y data
                (void *)dy.begin(),               // dy data
                xDataDesc,                        // descriptor for x data
                (void *)dx.begin(),               // dx data
                hx_desc_.desc(),                  // descriptor for hx data
                (void *)hx.begin(),               // hx data
                (void *)dhy.begin(),              // dhy data
                (void *)dhx.begin(),              // dhx data
                cx_desc_.desc(),                  // descriptor for cx data
                (void *)cx.begin(),               // cx data
                (void *)dcy.begin(),              // dcy data
                (void *)dcx.begin(),              // dcx data
                weight_size_,                     // weight space size
                (void *)weights_.begin(),         // weights
                workspace_size_,                  // workspace size
                (void *)workspace_.begin(),       // workspace
                train_size_,                      // reserve space size
                (void *)trainspace_.begin()       // reserve space
            ));
            
            // Clean up descriptors
            CHECK_CUDNN_ERROR(cudnnDestroyRNNDataDescriptor(yDataDesc));
            CHECK_CUDNN_ERROR(cudnnDestroyRNNDataDescriptor(xDataDesc));
            delete[] hostSeqLengths;
            cudaFree(devSeqLengths);
        }
};

template <typename T>
std::tuple<int, int> time_rnn(int hidden_size,
                              int batch_size,
                              int time_steps,
                              const std::string& type,
                              int inference) {

    cudnnRNN<T> rnn(hidden_size, batch_size, time_steps, type);

    auto x  = rand<T>({hidden_size, batch_size * time_steps}, curand_gen);
    auto y  = rand<T>({hidden_size, batch_size * time_steps}, curand_gen);
    auto dx = rand<T>({hidden_size, batch_size * time_steps}, curand_gen);
    auto dy = rand<T>({hidden_size, batch_size * time_steps}, curand_gen);

    auto hx = rand<T>({hidden_size, batch_size}, curand_gen);
    auto hy = rand<T>({hidden_size, batch_size}, curand_gen);
    auto cx = rand<T>({hidden_size, batch_size}, curand_gen);
    auto cy = rand<T>({hidden_size, batch_size}, curand_gen);
    auto dhx = rand<T>({hidden_size, batch_size}, curand_gen);
    auto dhy = rand<T>({hidden_size, batch_size}, curand_gen);
    auto dcx = rand<T>({hidden_size, batch_size}, curand_gen);
    auto dcy = rand<T>({hidden_size, batch_size}, curand_gen);

    int numRepeats = 1;

    //Warm up
    rnn.forward(x, hx, cx, y, hy, cy);

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
        rnn.forward(x, hx, cx, y, hy, cy);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    auto forward_time = std::chrono::duration<double, std::micro>(end - start).count() / numRepeats;
    int backward_time = 0;

    if (!inference) {
        //Warm up
        rnn.backward_data(y, dy, dhy, dcy,
                          hx, cx, dx, dhx, dcx);

        cudaDeviceSynchronize();

        start = std::chrono::steady_clock::now();

        for (int i = 0; i < numRepeats; ++i) {
            rnn.backward_data(y, dy, dhy, dcy,
                              hx, cx, dx, dhx, dcx);
        }
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();
        backward_time = std::chrono::duration<double, std::micro>(end - start).count() / numRepeats;

    }

    return std::make_tuple(static_cast<int>(forward_time),
                           static_cast<int>(backward_time));

}

int main(int argc, char **argv) {
    cudaFree(0);
    CHECK_CUDNN_ERROR( cudnnCreate(&cudnn_handle) );

    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    int inference = 0;

    if (argc > 1) {
        std::string inf = "inference";
        inference = argv[1] == inf ? 1 : 0;
    }

    std::vector<std::tuple<int, int, int, std::string>> dataset;

    if (argc > 3) {
        assert (argc == 7);
        dataset.push_back(
            std::make_tuple(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6])
        );
    }


#if CUDNN_MAJOR >= 6
    std::string precision;
    if (inference)
        precision = "int8";
    else
        precision = "half";
#else
    std::string precision = "float";
#endif
    if (argc > 2) {
        precision = argv[2];
    }

    if (inference) {
        std::cout << std::setw(45) << "Running inference benchmark " << std::endl;
    } else {
        std::cout << std::setw(45) << "Running training benchmark " << std::endl;
    }

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    type    hidden   N     timesteps   precision     fwd_time (usec)   ";
    if (!inference)
        std::cout << "bwd_time (usec)";
    std::cout << std::endl;
    for (const auto &problem : (dataset.empty() ? (inference ? inference_server_set : training_set) : dataset)) {
        int hidden_state, batch_size, time_steps;
        std::string type;
        std::tie(hidden_state, batch_size, time_steps, type) = problem;

        std::cout << std::setw(8) << type;
        std::cout << std::setw(8) << hidden_state;
        std::cout << std::setw(8) << batch_size;
        std::cout << std::setw(8) << time_steps;
        std::cout << std::setw(14) << precision;
        int fwd_time, bwd_time;

        std::stringstream ss;
        ss << "Unsupported precision requested. Precision: " << precision << " Inference: " << inference;

#if CUDNN_MAJOR >= 6
        if (inference) {
            if (precision == "float") {
                std::tie(fwd_time, bwd_time) = time_rnn<float>(hidden_state,
                                                               batch_size,
                                                               time_steps,
                                                               type,
                                                               inference);

            } else if (precision == "half") {
                std::tie(fwd_time, bwd_time) = time_rnn<uint16_t>(hidden_state,
                                                                  batch_size,
                                                                  time_steps,
                                                                  type,
                                                                  inference);
            } else if (precision == "int8") {
                std::tie(fwd_time, bwd_time) = time_rnn<uint8_t>(hidden_state,
                                                                 batch_size,
                                                                 time_steps,
                                                                 type,
                                                                 inference);
            } else {
                throw std::runtime_error(ss.str());
            }
        } else {
            if (precision == "float") {
                std::tie(fwd_time, bwd_time) = time_rnn<float>(hidden_state,
                                                               batch_size,
                                                               time_steps,
                                                               type,
                                                               inference);

            } else if (precision == "half") {
                std::tie(fwd_time, bwd_time) = time_rnn<uint16_t>(hidden_state,
                                                                  batch_size,
                                                                  time_steps,
                                                                  type,
                                                                  inference);
            } else {
                throw std::runtime_error(ss.str());
            }
        }
#else
        if (precision != "float")
            throw std::runtime_error(ss.str());
        std::tie(fwd_time, bwd_time) = time_rnn<float>(hidden_state,
                                                       batch_size,
                                                       time_steps,
                                                       type,
                                                       inference);
#endif

        std::cout << std::setw(18) << fwd_time;
        if (!inference)
            std::cout << std::setw(18) << bwd_time;
        std::cout << std::endl;
    }

    cudnnDestroy(cudnn_handle);
    curandDestroyGenerator(curand_gen);

    return 0;
}
