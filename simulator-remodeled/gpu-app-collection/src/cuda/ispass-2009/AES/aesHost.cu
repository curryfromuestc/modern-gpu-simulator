/***************************************************************************
 *   Copyright (C) 2006                                                    *
 *                                                                         *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


/**
	@author Svetlin Manavski <svetlin@manavski.com>
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

#include "sbox_E.h"
#include "sbox_D.h"
#include <aesEncrypt128_kernel.h>
#include <aesDecrypt128_kernel.h>
#include <aesEncrypt256_kernel.h>
#include <aesDecrypt256_kernel.h>

// Global texture objects
cudaTextureObject_t texEKey_obj = 0;
cudaTextureObject_t texDKey_obj = 0;
cudaTextureObject_t texEKey128_obj = 0;
cudaTextureObject_t texDKey128_obj = 0;

extern "C" void aesEncryptHandler128(unsigned *d_Result, unsigned *d_Input, int inputSize) {
    dim3  threads(BSIZE, 1);
    dim3  grid((inputSize/BSIZE)/4, 1);

    // Pass the texture object to the kernel
    aesEncrypt128<<< grid, threads >>>( d_Result, d_Input, inputSize, texEKey128_obj);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

extern "C" void aesDecryptHandler128(unsigned *d_Result, unsigned *d_Input, int inputSize) {
    dim3  threads(BSIZE, 1);
    dim3  grid((inputSize/BSIZE)/4, 1);

    // Pass the texture object to the kernel
    aesDecrypt128<<< grid, threads >>>( d_Result, d_Input, inputSize, texDKey128_obj);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

extern "C" void aesEncryptHandler256(unsigned *d_Result, unsigned *d_Input, int inputSize) {
    dim3  threads(BSIZE, 1);
    dim3  grid((inputSize/BSIZE)/4, 1);

    // Pass the texture object to the kernel
    aesEncrypt256<<< grid, threads >>>( d_Result, d_Input, inputSize, texEKey_obj);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

extern "C" void aesDecryptHandler256(unsigned *d_Result, unsigned *d_Input, int inputSize) {
    dim3  threads(BSIZE, 1);
    dim3  grid((inputSize/BSIZE)/4, 1);

    // Pass the texture object to the kernel
    aesDecrypt256<<< grid, threads >>>( d_Result, d_Input, inputSize, texDKey_obj);
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

extern "C" int aesHost(unsigned char* result, const unsigned char* inData, int inputSize, const unsigned char* key, int keySize, bool toEncrypt)
{
	if (inputSize < 256) 
		return -1;
	if (inputSize % 256 > 0) 
		return -11;
	if (keySize != 240 && keySize != 176) 
		return -2;
	if (!result || !inData || !key)
		return -3;

    int deviceCount;                                                         
    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                
    if (deviceCount == 0) {                                                  
        fprintf(stderr, "There is no device.\n");                            
        exit(EXIT_FAILURE);                                                  
    }                                                                        
    int dev;                                                                 
    for (dev = 0; dev < deviceCount; ++dev) {                                
        cudaDeviceProp deviceProp;                                           
        CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));   
        if (deviceProp.major >= 1)                                           
            break;                                                           
    }                                                                        
    if (dev == deviceCount) {                                                
        fprintf(stderr, "There is no device supporting CUDA.\n");            
        exit(EXIT_FAILURE);                                                  
    }                                                                        
    else                                                                     
        CUDA_SAFE_CALL(cudaSetDevice(dev));                                  


    // allocate device memory
    unsigned * d_Input;
    CUDA_SAFE_CALL( cudaMalloc((void**) &d_Input, inputSize) );

	// the size of the memory for the key must be equal to keySize (every thread copies one key byte to shared memory)
    unsigned * d_Key;
    CUDA_SAFE_CALL( cudaMalloc((void**) &d_Key, keySize) );

	unsigned int ext_timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&ext_timer));
    CUT_SAFE_CALL(cutStartTimer(ext_timer));

    // copy host memory to device
    CUDA_SAFE_CALL( cudaMemcpy(d_Input, inData, inputSize, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_Key, key, keySize, cudaMemcpyHostToDevice) );

    // Create resource descriptor
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_Key;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 0;
    resDesc.res.linear.desc.z = 0;
    resDesc.res.linear.desc.w = 0;
    resDesc.res.linear.sizeInBytes = keySize;
    
    // Create texture descriptor
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.readMode = cudaReadModeElementType;
    
    // Create texture objects
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texEKey_obj, &resDesc, &texDesc, NULL));
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texDKey_obj, &resDesc, &texDesc, NULL));
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texEKey128_obj, &resDesc, &texDesc, NULL));
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texDKey128_obj, &resDesc, &texDesc, NULL));

    // allocate device memory for result
    unsigned int size_Result = inputSize;
    unsigned * d_Result;
    CUDA_SAFE_CALL( cudaMalloc((void**) &d_Result, size_Result) );
	CUDA_SAFE_CALL( cudaMemset(d_Result, 0, size_Result) );
	

	unsigned int int_timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&int_timer));
    CUT_SAFE_CALL(cutStartTimer(int_timer));

	if (!toEncrypt) {	
		printf("\nDECRYPTION.....\n\n");
		if (keySize != 240)
			aesDecryptHandler128( d_Result, d_Input, inputSize);
		else
			aesDecryptHandler256( d_Result, d_Input, inputSize);
	} else {
		printf("\nENCRYPTION.....\n\n");
		if (keySize != 240)
			aesEncryptHandler128( d_Result, d_Input, inputSize);
		else
			aesEncryptHandler256( d_Result, d_Input, inputSize);
	}
	
	CUT_SAFE_CALL(cutStopTimer(int_timer));
    printf("GPU processing time: %f (ms)\n", cutGetTimerValue(int_timer));
    CUT_SAFE_CALL(cutDeleteTimer(int_timer));

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

    // copy result from device to host
    CUDA_SAFE_CALL(cudaMemcpy(result, d_Result, size_Result, cudaMemcpyDeviceToHost) );

    CUT_SAFE_CALL(cutStopTimer(ext_timer));
    printf("Total processing time: %f (ms)\n\n", cutGetTimerValue(ext_timer));
    CUT_SAFE_CALL(cutDeleteTimer(ext_timer));

    // cleanup memory
    CUDA_SAFE_CALL(cudaFree(d_Input));
    CUDA_SAFE_CALL(cudaFree(d_Key));
    CUDA_SAFE_CALL(cudaFree(d_Result));

    // Destroy texture objects
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texEKey_obj));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texDKey_obj));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texEKey128_obj));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texDKey128_obj));

    return 0;
}

