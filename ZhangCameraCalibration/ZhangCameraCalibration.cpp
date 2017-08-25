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
	/*Ū���C�@�T�Ϲ��A�q�������X���I�A�M��﨤�I�i��ȹϤ���T��*/
	string filePath;							// ��J��󧨥ؿ�
	string format = "JPG";						// ���w�榡
	vector<string> files;						// ����m�ΦW��
	vector<string> filenames;					// ���W��
	vector<string> successfilenames;			// ���I�������\�����W��
	double sq = 25;									// �Щw�����ؤo
	int szr = 6;									// �Щw���C���I��
	int szc = 9;									// �Щw���樤�I��
	cout << "Enter the folder directory : ";	// ��J�v����󧨦�m
	cin >> filePath;
	//cout << "Enter the image format (ex:.jpg) : ";	// ��J�v���榡
	//cin >> format;
	cout << "Please enter the square size of board (mm) : "; //��J�Щw�����ؤo
	cin >> sq;
	cout << "Please enter the size of board : " << endl;
	cout << "row : ";
	cin >> szr;
	cout << "column : ";
	cin >> szc;

	int image_count = 0;						// �Ϲ��ƶq
	int success_image_count = 0;				// ���\Ū���Ϲ����ƶq
	Size image_size;							// �Ϲ����ؤo
	Size board_size = Size(szr, szc);			// �Щw�O�W�C��B�C�����I��
	Size square_size = Size(sq, sq);			// ��ڴ��q�o�쪺�Щw�O�W�C�ӴѽL�檺�j�p
	vector<Point2f> image_points_buf;			// �w�s�C�T�Ϲ��W�˴��쪺���I
	vector<vector<Point2f>> image_points_seq;	// �O�s�˴��쪺�Ҧ����I

	getFiles(filePath, format, filenames, files);
	cout << endl << "There are " << files.size() << " images." << endl;

	string subfilePath = filePath + "\\caliberation";			//�Ыؤl��Ƨ�
	_mkdir(subfilePath.c_str());

	filePath += "\\";
	ofstream fout(subfilePath + "\\caliberation.txt");			// �ЫثO�s�Щw���G����r��

	for (int i = 0; i < files.size(); i++)
	{
		Mat imageInput = imread(files[i]);
		image_count++;

		if (image_count == 1)								//Ū�J�Ĥ@�i�Ϥ�������Ϲ��e����T  
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			cout << "Image width = " << image_size.width << "Pixels" << endl;
			cout << "Image height = " << image_size.height << "Pixels" << endl << endl;
			cout << "Start to get image corners..." << endl << endl;
		}
		cout << "No." << image_count << "\t" << filenames[i] << " start..." << endl;		// �Ω��[�������X  

																							/* �������I */
		bool getcorners;
		getcorners = findChessboardCorners(imageInput, board_size, image_points_buf, CALIB_CB_FAST_CHECK);		//����Ϲ��i��ֳt�M��
		//if (getcorners == 0)			//�Y�䤣��Ϲ��h�i��վ��A�M��@��
		//{
		//	getcorners = findChessboardCorners(imageInput, board_size, image_points_buf, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		//}

		if (getcorners == 0)
		{
			cout << "corner detect failed!" << endl;		//�䤣�쨤�I
			continue;
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);
			/* �ȹϤ���T�� */
			//find4QuadCornerSubpix(view_gray, image_points_buf, Size(11, 11));		//��ʴ��������I�i���T��  
			TermCriteria param(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.1);
			cornerSubPix(view_gray, image_points_buf, Size(5, 5), Size(-1, -1), param);
			image_points_seq.push_back(image_points_buf);							//�O�s�ȹϤ����I
			cout << "corner detect succeed!" << endl;
			successfilenames.push_back(filenames[i]);
			success_image_count++;
			/* �b�Ϲ��W��ܨ��I��m */
			drawChessboardCorners(imageInput, board_size, image_points_buf, true);	//�Ω�b�Ϥ����аO���I
			string findpoint = subfilePath + "\\";
			findpoint.append(filenames[i]);
			imwrite(findpoint, imageInput);											//�s���Хܫ᪺�Ϥ�           
		}
	}

	int total = image_points_seq.size();
	cout << endl << "The corners of " << total << " images was successfully detected." << endl << endl;
	int CornerNum = board_size.width*board_size.height;								//�C�i�Ϥ��W�`�����I��

	fout << "Image width  : " << image_size.width << endl;
	fout << "Image height : " << image_size.height << endl << endl;

	if (total < 3)
	{
		cout << "Successfully detected images must be greater than 2." << endl;
		return 0;
	}

	/*�H�U�O�ṳ���Щw*/
	cout << "Start calibration ..." << endl;
	/*�ѽL�T����T*/
	vector<vector<Point3f>> object_points;		// �O�s�Щw�O�W���I���T���y��
												/*���~�Ѽ�*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // �ṳ�����ѼƯx�}
	vector<int> point_counts;								// �C�T�Ϲ������I���ƶq  
	Mat distCoeffs = Mat(1, 4, CV_32FC1, Scalar::all(0));	// �۾���5�ӷ��ܫY�ơGk1,k2,p1,p2
	vector<Mat> tvecsMat;									// �C�T�Ϲ�������V�q
	vector<Mat> rvecsMat;									// �C�T�Ϲ��������V�q

															/* ��l�ƼЩw�O�W���I���T���y�� */
	int i, j, t;
	for (t = 0; t < success_image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++)
		{
			for (j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				/* ���]�Щw�O��b�@�ɧ��Шt��z=0�������W */
				realPoint.x = i*square_size.width;
				realPoint.y = j*square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* ��l�ƨC�T�Ϲ��������I�ƶq�A���w�C�T�Ϲ������i�H�ݨ짹�㪺�Щw�O */
	for (i = 0; i < success_image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	/* �}�l�Щw */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat);
	cout << "Calibration done!" << endl << endl;
	/* ��Щw���G�i����� */
	cout << "Start to evaluate the calibration results ..." << endl << endl;
	double total_err = 0.0;						// �Ҧ��Ϲ��������~�t���`�M
	double err = 0.0;							// �C�T�Ϲ��������~�t
	vector<Point2f> image_points2;				// �O�s���s�p��o�쪺��v�I
	cout << "The calibration error for each image : " << endl;
	fout << "The calibration error for each image : " << endl;
	for (i = 0; i < success_image_count; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];
		/* �q�L�o�쪺�ṳ�����~�ѼơA��Ŷ����T���I�i�歫�s��v�p��A�o��s����v�I */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		/* �p��s����v�I�M�ª���v�I�������~�t*/
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

		/*�Щw�ץ��᪺�Ӥ�*/
		Mat dist_image = imread(subfilePath + "\\" + successfilenames[i]);
		Mat undist_image;
		Mat mapx, mapy;
		initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), Mat(), image_size, CV_32FC1, mapx, mapy);
		remap(dist_image, undist_image, mapx, mapy, INTER_LINEAR);
		//string undistimage_path = subfilePath + "\\" + "undist_";
		//undistimage_path.append(successfilenames[i]);
		//imwrite(undistimage_path, undist_image);											
		imwrite(subfilePath + "\\" + successfilenames[i], undist_image);				//�s���ե��᪺�Ϥ�    
	}
	std::cout << endl << "The average of re-projection error : " << total_err / success_image_count << " Pixels" << endl;
	fout << "The average of re-projection error : " << total_err / success_image_count << " Pixels" << endl << endl;
	std::cout << "The assessment is completed!" << endl << endl;
	/*�O�s�w�е��G*/
	std::cout << "Start to save the calibration results..." << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // �O�s�C�T�Ϲ�������x�}
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
	fout << "k1�G" << distCoeffs.at<double>(0) << endl;
	fout << "k2�G" << distCoeffs.at<double>(1) << endl;
	fout << "p1�G" << distCoeffs.at<double>(2) << endl;
	fout << "p2�G" << distCoeffs.at<double>(3) << endl << endl;

	for (int i = 0; i < success_image_count; i++)
	{
		/* �N����V�q�ഫ���۹���������x�} */
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


/*����S�w�榡���ɮצW*/
//�Ĥ@�Ӭ����|�r��(string����) 
//�ĤG�ӰѼƬ����w�榡(string����) 
//�ĤT�ӰѼƬ��ɮצW�٦s�x�ܼ�(vector����,�ޥζǻ�)
//�ĥ|�ӰѼƬ��ɮצ�m�s�x�ܼ�(vector����,�ޥζǻ�)

void getFiles(string path, string format, vector<string>& filenames, vector<string>& files)
{
	//�ɮױ���N�X    
	intptr_t   hFile = 0;
	//�ɸ�T    
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
