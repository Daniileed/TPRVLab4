
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include "omp.h"
using namespace std;
const int N = 4096;
double timerCuda = 0;
cudaError_t multiWithCuda(__int32* a, __int32* b, __int32* cCuda, size_t size);
__int32 matrixMultSc(__int32* mFirst, __int32* mSecond, __int32* mResult);
void matrixPrint(__int32* matrix);
__int32 matrixReset(__int32* matrix);
bool matrixComp(__int32* a, __int32* b);
__global__ void simpleMultiply(__int32* a, __int32* b, __int32* c, int N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	__int32 sum = 0;
	for (int k = 0; k < N; k++) {
		sum += a[row * N + k] * b[k * N + col];
	}
	c[row * N + col] = sum;
}
int main()
{
	// Задание двумерных массивов через указатели
	__int32* a = new __int32[N * N];
	__int32* b = new __int32[N * N];
	__int32* cCuda = new __int32[N * N];
	__int32* cDefault = new __int32[N * N];
	// Заполняем матрицы порядковычи числами (в первой всё по порядку с 0, во второй с 10), а результирующую зануляем
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i * N + j] = rand() % 5;
			b[i * N + j] = rand() % 5;
			cCuda[i * N + j] = 0;
			cDefault[i * N + j] = 0;
		}
	}
	cout << "<<<NO_CUDA>>>" << endl;
	double start = omp_get_wtime();
	matrixMultSc(a, b, cDefault);
	double end = omp_get_wtime();
	cout << "Time taken: " << end - start << " sec" << endl;
	cout << "<<<CUDA>>>" << endl;
	multiWithCuda(a, b, cCuda, N * N * sizeof(__int32));
	cout << "Time ratio NO_CUDA / CUDA: " << ((end - start) * 1000) / timerCuda <<
		endl;

	if (matrixComp(cCuda, cDefault))
	{
		cout << "Matrices are the same!" << endl;
	}
	else {
		cout << "Matrices AREN't the same!!!" << endl;
	}
	//matrixPrint(cCuda);
	//matrixPrint(cDefault);
	cout << "DONE" << endl;
}
// Функция скалярного перемножения матриц
__int32 matrixMultSc(__int32* mFirst, __int32* mSecond, __int32* mResult)
{
	double start = omp_get_wtime();
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < N; k++)
			{
				mResult[i * N + j] += mFirst[i * N + k] * mSecond[k * N + j];
			}
		}
	}
	double end = omp_get_wtime();
	return (end - start);
}
// Функция вывода матрицы
void matrixPrint(__int32* matrix)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			cout << matrix[i * N + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}
// Функция обнуления матрицы
__int32 matrixReset(__int32* matrix)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			matrix[i * N + j] = 0;
		}
	}
	return *matrix;
}
// Функция сравнения матриц
bool matrixComp(__int32* a, __int32* b)
{
	bool if_equal = true;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (a[i * N + j] != b[i * N + j])
			{
				if_equal = false;
				break;
			}
		}
	}
	return if_equal;
}
cudaError_t multiWithCuda(__int32* a, __int32* b, __int32* cCuda, size_t numBytes) {
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	__int32* dev_a, * dev_b, * dev_c; // device copy of a,b,cCuda
	// Allocate memory for the device
	cudaMalloc((void**)&dev_a, numBytes);
	cudaMalloc((void**)&dev_b, numBytes);
	cudaMalloc((void**)&dev_c, numBytes);
	//matrixPrint(a);
	//matrixPrint(b);
	// kernel configuration
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	// Cuda event listener
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Copy values to the device
	cudaEventRecord(start, 0);
	cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_c, cCuda, numBytes, cudaMemcpyHostToDevice);
	simpleMultiply << <numBlocks, threadsPerBlock >> > (dev_a, dev_b, dev_c, N);
	cudaMemcpy(cCuda, dev_c, numBytes, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	timerCuda = gpuTime;
	cout << "Time taken on GPU with blocksize " << threadsPerBlock.x << "x" <<
		threadsPerBlock.y <<
		" and number of blocks " << numBlocks.x << "x" << numBlocks.y << ": " <<
		gpuTime << " ms" << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return cudaStatus;
}
