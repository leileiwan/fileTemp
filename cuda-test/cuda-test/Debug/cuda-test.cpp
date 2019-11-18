// cuda-test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <memory.h>

#include "cublas_v2.h"
#include "cuda_runtime.h"

#define IDX2C(i, j, ld) ((j)*(ld) +(i))
#define CHECK_CUBLAS_ERROR(state) if(CUBLAS_STATUS_SUCCESS != state) \
printf("ERROR state %d in file %s at line %d.\n", state, __FILE__, __LINE__);

void matrixAdd(float* dst, float* A, float* x, float* y, int rows, int cols) {
	float alpha = 1.0f;
	float beta = 0.0f;
	float *dev_mat1 = 0, *dev_x = 0, *dev_y = 0;
	CHECK_CUBLAS_ERROR(cudaMalloc((void**)&dev_mat1, (rows*cols) * sizeof(float)));
	CHECK_CUBLAS_ERROR(cudaMalloc((void**)&dev_x, (cols) * sizeof(float)));
	CHECK_CUBLAS_ERROR(cudaMalloc((void**)&dev_y, (rows) * sizeof(float)));
	CHECK_CUBLAS_ERROR(cublasSetMatrix(rows, cols, sizeof(float), A, rows, dev_mat1, rows));
	CHECK_CUBLAS_ERROR(cublasSetVector(cols, sizeof(float), x, 1, dev_x, 1));

	cublasHandle_t matAddHandle;
	cublasCreate(&matAddHandle);

	CHECK_CUBLAS_ERROR(cublasSgemv_v2(matAddHandle, CUBLAS_OP_N, rows, cols,
		&alpha, dev_mat1, rows, dev_x, 1, &beta, dev_y, 1));

	CHECK_CUBLAS_ERROR(cublasGetVector(rows, sizeof(float), dev_y, 1, dst, 1));
	CHECK_CUBLAS_ERROR(cublasDestroy(matAddHandle));
	CHECK_CUBLAS_ERROR(cudaFree(dev_y));
	CHECK_CUBLAS_ERROR(cudaFree(dev_x));
	CHECK_CUBLAS_ERROR(cudaFree(dev_mat1));
}

void gen_init_matrix(float* dst, int rows, int cols, float min_val = 0, float max_val = 1) {

	if (NULL == dst)
		exit(-1);
	for (int r = 0; r < rows*cols; ++r) {
		dst[r] = (1.0f * (rand()) / RAND_MAX) *(max_val - min_val) + min_val;
	}
}

void trans_mat(float* mat, int rows, int cols) {
	float* tmp = (float*)(malloc)(rows*cols * sizeof(float));
	memcpy((void*)tmp, (void*)mat, rows*cols * sizeof(float));
	int count = 0;
	for (int i = 0; i < cols; ++i) {
		for (int j = 0; j < rows; ++j) {
			mat[count++] = tmp[j*cols + i];
		}
	}
	free(tmp);
}


void test() {
	int rows = 10, cols = 9;
	float *mat1 = 0, *mat2 = 0, *mat3 = 0;
	mat1 = (float*)malloc(rows*cols*(sizeof(float)));
	mat2 = (float*)malloc(cols * sizeof(float));
	mat3 = (float*)malloc(rows * sizeof(float));
	gen_init_matrix(mat1, rows, cols, 0, 1);
	gen_init_matrix(mat2, cols, 1, 0, 1);
	gen_init_matrix(mat3, rows, 1, 0, 1);

	trans_mat(mat1, rows, cols);
	matrixAdd(mat3, mat1, mat2, mat3, rows, cols);

	for (int i = 0; i < rows; ++i)
		printf("%f  ", mat3[i]);
	printf("\n");

	free(mat1);
	free(mat2);
	free(mat3);
}


int main() {
	srand(20161003);
	test();
	return 0;
}
