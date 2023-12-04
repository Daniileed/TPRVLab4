
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cmath>
#include <stdio.h>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define BLOCK_SIZE 32
#define Height 768
#define Length 1024
string filename = "C://Users//danii//source//repos//TPRVLab4//" + std::to_string(Length) + "x" + std::to_string(Height) + ".jpg";
#define output_path "C://Users//danii//source//repos//TPRVLab4//result.jpg"

__global__ void imageProcessKernel(const int* Iv, const int* FSX, const int* FSY, int* MX, int* MY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < Length && idy < Height) {
        int index = idx * Height + idy;
        if (index - Length - 1 >= 0 && index - Length - 1 < Length*Height)
            MX[index] += Iv[index-Length - 1] * FSX[0];
        if (index - Length + 1 >= 0 && index - Length + 1 < Length * Height)
            MX[index] += Iv[index - Length + 1] * FSX[2];
        if (index - 1 >= 0 && index - 1 < Length * Height)
            MX[index] += Iv[index - 1] * FSX[3];
        if (index + 1 >= 0 && index + 1 < Length * Height)
            MX[index] += Iv[index + 1] * FSX[5];
        if (index + Length - 1 >= 0 && index + Length - 1 < Length * Height)
            MX[index] += Iv[index + Length - 1] * FSX[6];
        if (index + Length + 1 >= 0 && index + Length + 1 < Length * Height)
            MX[index] += Iv[index + Length + 1] * FSX[8];

        if (index - Length - 1 >= 0 && index - Length - 1 < Length * Height)
            MY[index] += Iv[index - Length - 1] * FSY[0];
        if (index - Length >= 0 && index - Length < Length * Height)
            MY[index] += Iv[index - Length] * FSY[1];
        if (index - Length + 1 >= 0 && index - Length + 1 < Length * Height)
            MY[index] += Iv[index - Length + 1] * FSY[2];
        if (index + Length - 1 >= 0 && index + Length - 1 < Length * Height)
            MY[index] += Iv[index + Length - 1] * FSY[6];
        if (index + Length >= 0 && index + Length < Length * Height)
            MY[index] += Iv[index + Length] * FSY[7];
        if (index + Length + 1 >= 0 && index + Length + 1 < Length * Height)
            MY[index] += Iv[index + Length + 1] * FSY[8];
    }
}

__global__ void meanSquare(int* MX, int* MY, int* MRv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < Length && idy < Height) {
        int index = idx * Height + idy;
        MRv[index] = floor(sqrt(pow(MX[index], 2) + pow(MY[index], 2)));
    }
}

int Rv[Height][Length];
int Gv[Height][Length];
int Bv[Height][Length];
int Iv[Height][Length];

int MRv[Height][Length];
int MX[Height][Length];
int MY[Height][Length];

int MRMax = 1;

const int FSX[3][3] = {
{ -1,0,1 },
    {-2,0,2},
{-1,0,1 }
};

const int FSY[3][3] = {
    {-1,-2,-1},
    {0,0,0},
    {1,2,1 } };



void check_elementX(int curI, int curJ, int i, int j, int elX, int elY) {
    if ((i >= 0) && (i < Height) && (j >= 0) && (j < Length))
        MX[curI][curJ] += Iv[i][j] * FSX[elX][elY];
}

void check_elementY(int curI, int curJ, int i, int j, int elX, int elY) {
    if ((i >= 0) && (i < Height) && (j >= 0) && (j < Length))
        MY[curI][curJ] += Iv[i][j] * FSY[elX][elY];
}

int main()
{
    int numBytes = Length * Height * sizeof(int);

    int* Iv_dev = NULL;
    int* FSX_dev = NULL;
    int* FSY_dev = NULL;
    int* MX_dev = NULL;
    int* MY_dev = NULL;
    int* MRv_dev = NULL;
    int  MRMax_dev = 1;

    Mat img_color = imread(filename);
    Mat3i result_img(Height, Length);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            Bv[i][j] = img_color.at<cv::Vec3b>(i, j)[0];
            Gv[i][j] = img_color.at<cv::Vec3b>(i, j)[1];
            Rv[i][j] = img_color.at<cv::Vec3b>(i, j)[2];
            Iv[i][j] = floor((Rv[i][j] + Gv[i][j] + Bv[i][j]) / 3);
        }
    }


    //CUDA
    cudaMalloc((void**)&Iv_dev, numBytes);
    cudaMalloc((void**)&FSX_dev, sizeof(int)*9);
    cudaMalloc((void**)&FSY_dev, sizeof(int)*9);

    cudaMalloc((void**)&MX_dev, numBytes);
    cudaMalloc((void**)&MY_dev, numBytes);
    cudaMalloc((void**)&MRv_dev, numBytes);

    cudaMemcpy(Iv_dev, Iv, numBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(FSX_dev, FSX, sizeof(int) * 9, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(FSY_dev, FSY, sizeof(int) * 9, cudaMemcpyKind::cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(Length / threads.x, Height / threads.y);

    imageProcessKernel << <blocks, threads >>> (Iv_dev, FSX_dev, FSY_dev, MX_dev, MY_dev);

    cudaMemcpy(MRv_dev, MRv, numBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

    meanSquare << <blocks, threads >> > (MX_dev, MY_dev, MRv_dev);

    cudaMemcpy(MRv, MRv_dev, numBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);




    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(Iv_dev);
        cudaFree(FSX_dev);
        cudaFree(FSY_dev);
        cudaFree(MX_dev);
        cudaFree(MY_dev);
        cudaFree(MRv_dev);
        return 1;
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            if (MRv[i][j] >= MRMax)
                MRMax = MRv[i][j];
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            MRv[i][j] = MRv[i][j] * 255 / MRMax;
        }
    }

    Mat Sobel_scale = Mat::zeros(Height, Length, CV_8UC1);

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++)
        {
            Sobel_scale.at<uchar>(i, j) = MRv[i][j];
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    imwrite(output_path, Sobel_scale);
    cout << "Process finished! " << endl;
    cout << "Time spent executing by the GPU: " << gpuTime / 1000 << " seconds" << endl;
    

    Err:
    cudaFree(Iv_dev);
    cudaFree(FSX_dev);
    cudaFree(FSY_dev);
    cudaFree(MX_dev);
    cudaFree(MY_dev);
    cudaFree(MRv_dev);

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            Iv[i][j] = 0;
            MX[i][j] = 0;
            MY[i][j] = 0;
            MRv[i][j] = 0;
        }
    }

     Sobel_scale = Mat::zeros(Height, Length, CV_8UC1);

    auto start_1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            Bv[i][j] = img_color.at<cv::Vec3b>(i, j)[0];
            Gv[i][j] = img_color.at<cv::Vec3b>(i, j)[1];
            Rv[i][j] = img_color.at<cv::Vec3b>(i, j)[2];
            Iv[i][j] = floor((Rv[i][j] + Gv[i][j] + Bv[i][j]) / 3);
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            check_elementX(i, j, i - 1, j - 1, 0, 0);
            check_elementX(i, j, i, j - 1, 1, 0);
            check_elementX(i, j, i + 1, j - 1, 2, 0);

            check_elementX(i, j, i - 1, j + 1, 0, 2);
            check_elementX(i, j, i, j + 1, 1, 2);
            check_elementX(i, j, i + 1, j + 1, 2, 2);

            check_elementY(i, j, i - 1, j - 1, 0, 0);
            check_elementY(i, j, i - 1, j, 0, 1);
            check_elementY(i, j, i - 1, j + 1, 0, 2);

            check_elementY(i, j, i + 1, j - 1, 2, 0);
            check_elementY(i, j, i + 1, j, 2, 1);
            check_elementY(i, j, i + 1, j + 1, 2, 2);

        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            MRv[i][j] = floor(sqrt(pow(MX[i][j], 2) + pow(MY[i][j], 2)));
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            if (MRv[i][j] >= MRMax)
                MRMax = MRv[i][j];
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            MRv[i][j] = MRv[i][j] * 255 / MRMax;
        }
    }

    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Length; j++) {
            Sobel_scale.at<uchar>(i, j) = MRv[i][j];
        }
    }


    auto end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> dur1 = (end_1 - start_1);
    std::cout << "Procesor Time: " << dur1.count() << " seconds\n\n";

  //  namedWindow("Sobel_scale", WINDOW_NORMAL);
  //  imshow("Sobel_scale", Sobel_scale);
  //  waitKey(0);
  //  destroyAllWindows();

    return 0;
}
