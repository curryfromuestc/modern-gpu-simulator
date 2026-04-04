////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mergesort.cuh"
#include "mergesort_kernel.cu"
////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE	256
#define ROW_LENGTH	BLOCKSIZE * 4
#define ROWS		4096

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////
float4* runMergeSort(int listsize, int divisions, 
				     float4 *d_origList, float4 *d_resultList, 
				     int *sizes, int *nullElements,
					 unsigned int *origOffsets)
{
	int *startaddr = (int *)malloc((divisions + 1)*sizeof(int)); 
	int largestSize = -1; 
	startaddr[0] = 0; 
	for(int i=1; i<=divisions; i++)
	{
		startaddr[i] = startaddr[i-1] + sizes[i-1];
		if(sizes[i-1] > largestSize) largestSize = sizes[i-1]; 
	}
	largestSize *= 4; 

	// Setup texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	
	// Create resource descriptor
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = d_origList;
	resDesc.res.linear.desc = channelDesc;
	resDesc.res.linear.sizeInBytes = listsize * sizeof(float);
	
	// Create texture descriptor
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;
	
	// Create texture object
	cudaTextureObject_t tex_obj = 0;
	cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL);

	////////////////////////////////////////////////////////////////////////////
	// First sort all float4 elements internally
	////////////////////////////////////////////////////////////////////////////
	#ifdef MERGE_WG_SIZE_0
	const int THREADS = MERGE_WG_SIZE_0;
	#else
	const int THREADS = 256; 
	#endif
	dim3 threads(THREADS, 1);
	int blocks = ((listsize/4)%THREADS == 0) ? (listsize/4)/THREADS : (listsize/4)/THREADS + 1; 
	dim3 grid(blocks, 1);
	mergeSortFirst<<< grid, threads >>>(d_resultList, listsize, tex_obj);

	////////////////////////////////////////////////////////////////////////////
	// Then, go level by level
	////////////////////////////////////////////////////////////////////////////
	cudaMemcpyToSymbol(constStartAddr, startaddr, (divisions + 1)*sizeof(int)); 
	cudaMemcpyToSymbol(finalStartAddr, origOffsets, (divisions + 1)*sizeof(int)); 
	cudaMemcpyToSymbol(nullElems, nullElements, (divisions)*sizeof(int)); 
	int nrElems = 2;
	while(true){
		int floatsperthread = (nrElems*4); 
		int threadsPerDiv = (int)ceil(largestSize/(float)floatsperthread); 
		int threadsNeeded = threadsPerDiv * divisions; 
		#ifdef MERGE_WG_SIZE_1
		threads.x = MERGE_WG_SIZE_1;
		#else
		threads.x = 208; 
		#endif
		grid.x = ((threadsNeeded%threads.x) == 0) ?
			threadsNeeded/threads.x : 
			(threadsNeeded/threads.x) + 1; 
		if(grid.x < 8){
			grid.x = 8; 
			threads.x = ((threadsNeeded%grid.x) == 0) ? 
				threadsNeeded / grid.x : 
				(threadsNeeded / grid.x) + 1; 
		}
		// Swap orig/result list
		float4 *tempList = d_origList; 
		d_origList = d_resultList; 
		d_resultList = tempList; 
		
		// Update texture object with new data
		cudaDestroyTextureObject(tex_obj);
		resDesc.res.linear.devPtr = d_origList;
		cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL);
		
		mergeSortPass <<< grid, threads >>>(d_resultList, nrElems, threadsPerDiv, tex_obj); 
		nrElems *= 2; 
		floatsperthread = (nrElems*4); 
		if(threadsPerDiv == 1) break; 
	}
	////////////////////////////////////////////////////////////////////////////
	// Now, get rid of the NULL elements
	////////////////////////////////////////////////////////////////////////////
	#ifdef MERGE_WG_SIZE_0
	threads.x = MERGE_WG_SIZE_0;
	#else
	threads.x = 256; 
	#endif
	grid.x = ((largestSize%threads.x) == 0) ?
			largestSize/threads.x : 
			(largestSize/threads.x) + 1; 
	grid.y = divisions; 
	mergepack <<< grid, threads >>> ((float *)d_resultList, (float *)d_origList);

	// Clean up texture object
	cudaDestroyTextureObject(tex_obj);
	
	free(startaddr);
	return d_origList; 
}
