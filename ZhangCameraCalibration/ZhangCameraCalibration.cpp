// homework01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <io.h>
#include <direct.h>
#include <string>  
#include <vector>

using namespace cv;
using namespace std;

void getFiles(string path, string format, vector<string>& filenames, vector<string>& files);

int main()
{
	/*讀取每一幅圖像，從中提取出角點，然後對角點進行亞圖元精確化*/
	string filePath;							// 輸入文件夾目錄
	string format = "JPG";						// 指定格式
	vector<string> files;						// 文件位置及名稱
	vector<string> filenames;					// 文件名稱
	vector<string> successfilenames;			// 角點偵測成功的文件名稱
	double sq = 25;									// 標定版方格尺寸
	int szr = 6;									// 標定版列角點數
	int szc = 9;									// 標定版行角點數
	cout << "Enter the folder directory : ";	// 輸入影像文件夾位置
	cin >> filePath;
	//cout << "Enter the image format (ex:.jpg) : ";	// 輸入影像格式
	//cin >> format;
	cout << "Please enter the square size of board (mm) : "; //輸入標定版方格尺寸
	cin >> sq;
	cout << "Please enter the size of board : " << endl;
	cout << "row : ";
	cin >> szr;
	cout << "column : ";
	cin >> szc;

	int image_count = 0;						// 圖像數量
	int success_image_count = 0;				// 成功讀取圖像的數量
	Size image_size;							// 圖像的尺寸
	Size board_size = Size(szr, szc);			// 標定板上每行、列的角點數
	Size square_size = Size(sq, sq);			// 實際測量得到的標定板上每個棋盤格的大小
	vector<Point2f> image_points_buf;			// 緩存每幅圖像上檢測到的角點
	vector<vector<Point2f>> image_points_seq;	// 保存檢測到的所有角點

	getFiles(filePath, format, filenames, files);
	cout << endl << "There are " << files.size() << " images." << endl;

	string subfilePath = filePath + "\\caliberation";			//創建子資料夾
	_mkdir(subfilePath.c_str());

	filePath += "\\";
	ofstream fout(subfilePath + "\\caliberation.txt");			// 創建保存標定結果的文字檔

	for (int i = 0; i < files.size(); i++)
	{
		Mat imageInput = imread(files[i]);
		image_count++;

		if (image_count == 1)								//讀入第一張圖片時獲取圖像寬高資訊  
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			cout << "Image width = " << image_size.width << "Pixels" << endl;
			cout << "Image height = " << image_size.height << "Pixels" << endl << endl;
			cout << "Start to get image corners..." << endl << endl;
		}
		cout << "No." << image_count << "\t" << filenames[i] << " start..." << endl;		// 用於觀察檢驗輸出  

																							/* 提取角點 */
		bool getcorners;
		getcorners = findChessboardCorners(imageInput, board_size, image_points_buf, CALIB_CB_FAST_CHECK);		//先對圖像進行快速尋找
		//if (getcorners == 0)			//若找不到圖像則進行調整後再尋找一次
		//{
		//	getcorners = findChessboardCorners(imageInput, board_size, image_points_buf, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		//}

		if (getcorners == 0)
		{
			cout << "corner detect failed!" << endl;		//找不到角點
			continue;
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);
			/* 亞圖元精確化 */
			//find4QuadCornerSubpix(view_gray, image_points_buf, Size(11, 11));		//對粗提取的角點進行精確化  
			TermCriteria param(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.1);
			cornerSubPix(view_gray, image_points_buf, Size(5, 5), Size(-1, -1), param);
			image_points_seq.push_back(image_points_buf);							//保存亞圖元角點
			cout << "corner detect succeed!" << endl;
			successfilenames.push_back(filenames[i]);
			success_image_count++;
			/* 在圖像上顯示角點位置 */
			drawChessboardCorners(imageInput, board_size, image_points_buf, true);	//用於在圖片中標記角點
			string findpoint = subfilePath + "\\";
			findpoint.append(filenames[i]);
			imwrite(findpoint, imageInput);											//存取標示後的圖片           
		}
	}

	int total = image_points_seq.size();
	cout << endl << "The corners of " << total << " images was successfully detected." << endl << endl;
	int CornerNum = board_size.width*board_size.height;								//每張圖片上總的角點數

	fout << "Image width  : " << image_size.width << endl;
	fout << "Image height : " << image_size.height << endl << endl;

	if (total < 3)
	{
		cout << "Successfully detected images must be greater than 2." << endl;
		return 0;
	}

	/*以下是攝像機標定*/
	cout << "Start calibration ..." << endl;
	/*棋盤三維資訊*/
	vector<vector<Point3f>> object_points;		// 保存標定板上角點的三維座標
												/*內外參數*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 攝像機內參數矩陣
	vector<int> point_counts;								// 每幅圖像中角點的數量  
	Mat distCoeffs = Mat(1, 4, CV_32FC1, Scalar::all(0));	// 相機的5個畸變係數：k1,k2,p1,p2
	vector<Mat> tvecsMat;									// 每幅圖像的旋轉向量
	vector<Mat> rvecsMat;									// 每幅圖像的平移向量

															/* 初始化標定板上角點的三維座標 */
	int i, j, t;
	for (t = 0; t < success_image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				/* 假設標定板放在世界坐標系中z=0的平面上 */
				realPoint.x = i*square_size.width;
				realPoint.y = j*square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* 初始化每幅圖像中的角點數量，假定每幅圖像中都可以看到完整的標定板 */
	for (i = 0; i < success_image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	/* 開始標定 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat);
	cout << "Calibration done!" << endl << endl;
	/* 對標定結果進行評價 */
	cout << "Start to evaluate the calibration results ..." << endl << endl;
	double total_err = 0.0;						// 所有圖像的平均誤差的總和
	double err = 0.0;							// 每幅圖像的平均誤差
	vector<Point2f> image_points2;				// 保存重新計算得到的投影點
	cout << "The calibration error for each image : " << endl;
	fout << "The calibration error for each image : " << endl;
	for (i = 0; i < success_image_count; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];
		/* 通過得到的攝像機內外參數，對空間的三維點進行重新投影計算，得到新的投影點 */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		/* 計算新的投影點和舊的投影點之間的誤差*/
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		std::cout << successfilenames[i] << "\taverage error : " << err << " Pixels" << endl;
		fout << successfilenames[i] << "\taverage error : " << err << " Pixels" << endl;

		/*標定修正後的照片*/
		Mat dist_image = imread(subfilePath + "\\" + successfilenames[i]);
		Mat undist_image;
		Mat mapx, mapy;
		initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), Mat(), image_size, CV_32FC1, mapx, mapy);
		remap(dist_image, undist_image, mapx, mapy, INTER_LINEAR);
		//string undistimage_path = subfilePath + "\\" + "undist_";
		//undistimage_path.append(successfilenames[i]);
		//imwrite(undistimage_path, undist_image);											
		imwrite(subfilePath + "\\" + successfilenames[i], undist_image);				//存取校正後的圖片    
	}
	std::cout << endl << "The average of re-projection error : " << total_err / success_image_count << " Pixels" << endl;
	fout << "The average of re-projection error : " << total_err / success_image_count << " Pixels" << endl << endl;
	std::cout << "The assessment is completed!" << endl << endl;
	/*保存定標結果*/
	std::cout << "Start to save the calibration results..." << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 保存每幅圖像的旋轉矩陣
	fout << "Intrinsic parameter" << endl;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			fout << cameraMatrix.at<double>(i, j) << "\t";
		}
		fout << endl;
	}
	fout << endl;

	fout << "Distortion coefficient" << endl;
	fout << "k1：" << distCoeffs.at<double>(0) << endl;
	fout << "k2：" << distCoeffs.at<double>(1) << endl;
	fout << "p1：" << distCoeffs.at<double>(2) << endl;
	fout << "p2：" << distCoeffs.at<double>(3) << endl << endl;

	for (int i = 0; i < success_image_count; i++)
	{
		/* 將旋轉向量轉換為相對應的旋轉矩陣 */
		Rodrigues(rvecsMat[i], rotation_matrix);
		fout << successfilenames[i] << "\tExtrinsic parameter" << endl;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				fout << rotation_matrix.at<double>(i, j) << "\t";
			}
			fout << tvecsMat[i].at<double>(i) << "\t";
			fout << endl;
		}
		fout << endl;
	}
	fout.close();
	std::cout << "Results are saved!" << endl;
	return 0;
}


/*獲取特定格式的檔案名*/
//第一個為路徑字串(string類型) 
//第二個參數為指定格式(string類型) 
//第三個參數為檔案名稱存儲變數(vector類型,引用傳遞)
//第四個參數為檔案位置存儲變數(vector類型,引用傳遞)

void getFiles(string path, string format, vector<string>& filenames, vector<string>& files)
{
	//檔案控制代碼    
	intptr_t   hFile = 0;
	//檔資訊    
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					getFiles(p.assign(path).append("\\").append(fileinfo.name), format, filenames, files);
				}
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				filenames.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}
