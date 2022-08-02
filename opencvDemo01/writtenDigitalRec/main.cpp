#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>
#include"thin.h"
using namespace std;
using namespace cv;

Mat imgSrc, imgSrcP, imgTem, imgTemP;
vector<Mat> nums;
//vector<Mat> fnums;
vector<Point2f> pts;
vector<Point2f> cpt;
int temp[8] = { 0 };
float templateMatchSQDIFF(int* templ, int* toBeMatch) {
    float result = 0;
    float plus1 = 0;
    float plus2 = 0;
    for (int i = 0; i < 8; i++) {
        result += pow(templ[i] - toBeMatch[i], 2);
        plus1 += pow(templ[i], 2);
        plus2 += pow(toBeMatch[i], 2);
    }

    return result/sqrt(plus1*plus2);
}

float templateMatchCCORR(int* templ, int* toBeMatch) {
    int result = 0;
    int t1 = 0;
    int t2 = 0;
    for (int i = 0; i < 8; i++) {
        result += templ[i] * toBeMatch[i];
        t1 += pow(temp[i], 2);
        t2 += pow(toBeMatch[i], 2);
    }
    return result/sqrt(t1*t2);
}

void readFeature(int* toBeMatch) {
    string filename;
    string dir;
    string finalDir;
    string num;
    int length;
    int p;
    int  count = 0;
    float min = 999999;
    float max = 0;
    int result1 = 0;
    int result2 = 0;
    int numb = 0;
    int avg = 0;
    for (int i = 0; i <= 9; i++) {
        ifstream infile;
        filename = to_string(i).append(".txt");
        dir = "writtenFeatureVecs/";
        finalDir = "";
        finalDir = dir.append(filename);
        infile.open(finalDir, ios::in);
        string buf;
        avg = 0;
        numb = 0;
        while (getline(infile, buf)) {
            numb++;
            length = buf.length();
            p = 0;
            num = "";
            count = 0;
            
            while (p<length) {
                
                if (buf[p] == ' ') {
                    
                    temp[count] = atoi(num.c_str());
                    num = "";
                    p++;
                    count++;
                }
                else if (p == length - 1) {
                    num.append(to_string(buf[p]-48));
             
                    temp[count] = atoi(num.c_str());
                    num = "";
                    p++;
                }else {
                    num.append(to_string(buf[p]-48));
                    p++;
                }
            }
            
            /*if (min > templateMatchSQDIFF(temp, toBeMatch)) {
                min = templateMatchSQDIFF(temp, toBeMatch);
                result1 = i;
            }*/
            if (max < templateMatchCCORR(temp, toBeMatch)) {
                max = templateMatchCCORR(temp, toBeMatch);
                
                result2 = i;
            }

        }
        infile.close();
        

            /*if (max < avg) {
                max = avg;
                result2 = i;
            }*/
    }
    
    //cout <<"using SQDIFF: " <<result1 << " min = "<< min <<endl;
    cout <<"using CCORR: " << result2 <<" max = "<<max<< endl;
    cout << endl;
}

void showPos(Mat img, string s, Point p) {
    circle(img, p, 4, Scalar(255, 0, 0), -1);
    putText(img, s, p, cv::FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0));
}

void showFrame(Mat img, int width, int height) {
    Point start = Point(0, height / 2);
    Point end = Point(width, height / 2);
    line(img, start, end, cv::Scalar(255, 0, 0));

    start = Point(width / 2, 0);
    end = Point(width / 2, height);
    line(img, start, end, cv::Scalar(255, 0, 0));

    start = Point(0, 0);
    end = Point(width, height);
    line(img, start, end, cv::Scalar(255, 0, 0));

    start = Point(0 , width);
    end = Point(height, 0);
    line(img, start, end, cv::Scalar(255, 0, 0));

    
}

void saveFeatureVec(int* vec, int number) {
    string targetTxt = to_string(number).append(".txt");
    string finalDir = "writtenFeatureVecs/";
    ofstream OutFile(finalDir.append(targetTxt), ios::app);

    for (int i = 0; i < 8; i++) {
        OutFile << vec[i] << " ";
    }
    OutFile << endl;

    OutFile.close();
}

void TemCalculation(int* locX, int* locY, int width, int height, int number) {
    int distance[8] = {0};
    int w = width / 2;
    int h = height / 2;
    for (int i = 0; i < 8; i++) {
        distance[i] = sqrt(pow((locX[i] - w), 2) + pow((locY[i] - h) , 2)) * 256 / width;
    }
   
    saveFeatureVec(distance, number);

}

void calculation(int* locX, int* locY, int width, int height, int number) {
    int distance[8] = { 0 };
    int w = width / 2;
    int h = height / 2;
    for (int i = 0; i < 8; i++) {
        distance[i] = sqrt(pow((locX[i] - w), 2) + pow((locY[i] - h), 2)) * 256 / width;
    }
    
    readFeature(distance);
}


void makeTemFeatureValue(Mat img, int number) {
    int width = img.cols;
    int height = img.rows;
    int locX[8];
    int locY[8];
    bool conf[8] = { false };
    int tp;
    
    for (int i = width - 1; i > width /2; i--) {
        
        if (!conf[0] && img.at<uchar>(height / 2, i) != 0) {
            locX[0] = i;
            locY[0] = height / 2;
            conf[0] = true;
        }
       
        if (!conf[1] && img.at<uchar>(height - i, i) != 0) {
            locX[1] = i;
            locY[1] = height - i;
            conf[1] = true;
            
        }

        if (!conf[6] && img.at<uchar>(i, width / 2) != 0) {
            locX[6] = width / 2;
            locY[6] = i;
            conf[6] = true;
        }

        if (!conf[7] && img.at<uchar>(i, i) != 0) {
            locX[7] = i;
            locY[7] = i;
            conf[7] = true;
        }
        
    }
   

    for (int i = 0 ; i < height / 2; i++) {
        if (!conf[2] && img.at<uchar>(i, height/2) != 0) {
            locX[2] = height / 2;
            locY[2] = i;
            conf[2] = true;
        }
       
        if (!conf[3] && img.at<uchar>(i, i) != 0) {
            locX[3] = i;
            locY[3] = i;
            conf[3] = true;

        }

        if (!conf[4] && img.at<uchar>(width / 2, i) != 0) {
            locX[4] = i;
            locY[4] = width / 2;
            conf[4] = true;
        }


        if (!conf[5] && img.at<uchar>(width - 1 - i, i) != 0) {
            locX[5] = i;
            locY[5] = width - 1 - i;
            conf[5] = true;
        }
    }
    
    string s;
    Point p;

    for (int i = 0; i < 8; i++) {
        if (!conf[i]) {
            locX[i] = width / 2;
            locY[i] = height / 2;
        }
        p.x = locX[i];
        p.y = locY[i];
        s = to_string(locX[i]) + ", " + to_string(locY[i]);
        showPos(img, s, p);

    }
    
    showFrame(img, width, height);

    //TemCalculation(locX, locY, width, height, number);
}

void makeFeatureValue(Mat img, int number) {
    int width = img.cols;
    int height = img.rows;
    int locX[8];
    int locY[8];
    bool conf[8] = { false };
    int tp;

    for (int i = width - 1; i > width / 2; i--) {

        if (!conf[0] && img.at<uchar>(height / 2, i) != 0) {
            locX[0] = i;
            locY[0] = height / 2;
            conf[0] = true;
        }

        if (!conf[1] && img.at<uchar>(height - i, i) != 0) {
            locX[1] = i;
            locY[1] = height - i;
            conf[1] = true;

        }

        if (!conf[6] && img.at<uchar>(i, width / 2) != 0) {
            locX[6] = width / 2;
            locY[6] = i;
            conf[6] = true;
        }

        if (!conf[7] && img.at<uchar>(i, i) != 0) {
            locX[7] = i;
            locY[7] = i;
            conf[7] = true;
        }

    }


    for (int i = 0; i < height / 2; i++) {
        if (!conf[2] && img.at<uchar>(i, height / 2) != 0) {
            locX[2] = height / 2;
            locY[2] = i;
            conf[2] = true;
        }

        if (!conf[3] && img.at<uchar>(i, i) != 0) {
            locX[3] = i;
            locY[3] = i;
            conf[3] = true;

        }

        if (!conf[4] && img.at<uchar>(width / 2, i) != 0) {
            locX[4] = i;
            locY[4] = width / 2;
            conf[4] = true;
        }


        if (!conf[5] && img.at<uchar>(width - 1 - i, i) != 0) {
            locX[5] = i;
            locY[5] = width - 1 - i;
            conf[5] = true;
        }
    }

    string s;
    Point p;

    for (int i = 0; i < 8; i++) {
        if (!conf[i]) {
            locX[i] = width / 2;
            locY[i] = height / 2;
        }
        p.x = locX[i];
        p.y = locY[i];
        s = to_string(locX[i]) + ", " + to_string(locY[i]);
        showPos(img, s, p);

    }

    showFrame(img, width, height);

    calculation(locX, locY, width, height, number);
}

void getContours(Mat imgD) {
	Mat imgNew;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgD, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> contPoly(contours.size());
	vector<Rect> boundRect(contours.size());
    int height, width, sub;
    vector<int> sizeX;
	for (int i = 0; i < contours.size(); i++) {
		if (contourArea(contours[i]) > 20) {
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], contPoly[i], 0.02 * peri, true);
			boundRect[i] = boundingRect(contPoly[i]);
			Rect roi (boundRect[i].x ,boundRect[i].y ,boundRect[i].width, boundRect[i].height);
			imgNew = imgD(roi);
            height = boundRect[i].height;
            width = boundRect[i].width;
            Point cp(width / 2, height / 2);
            cpt.push_back(cp);
			nums.push_back(imgNew);
            sizeX.push_back(boundRect[i].x);
		}
	}
    //imshow("contours", imgD);
    Mat imgT;
    int picXT;
    Point cpTp;
    for (int i = 0; i < contours.size(); i++) {
        for (int j = i + 1; j < contours.size(); j++) {
            if (sizeX[j] < sizeX[i]) {
                imgT = nums[j];
                picXT = sizeX[j];
                cpTp = cpt[j];
                nums[j] = nums[i];
                sizeX[j] = sizeX[i];
                cpt[j] = cpt[i];
                nums[i] = imgT;
                sizeX[i] = picXT;
                cpt[i] = cpTp;

            }
        }
    }

}

Mat preprocessingT(Mat imgTem) {

	Mat imgGray;
	cvtColor(imgTem, imgGray, COLOR_BGR2GRAY);
    
	Mat imgBin;
	threshold(imgGray, imgBin, 128, 255, THRESH_BINARY_INV); 
    
    //blur(imgBin, imgBin, Size(3, 3));
    imshow("0-1", imgBin);
	return imgBin;
}

Mat getImg(string path) {
	return (imread(path));
}

void resizePic() {
    
    int count = 0;
    Mat imgs;
    Moments m;
    int moveX;
    int moveY;
    int width;
    int height;
    string title;
    for (int i = 0; i < nums.size(); i++) {
        imgs = nums[i];
        threshold(imgs, imgs, 128, 1, THRESH_BINARY);
        thin th;
        imgs = th.thinImage(imgs, -1);
        imgs *= 255;
     
        m = moments(imgs, true);
        
        Point p(m.m10 / m.m00, m.m01 / m.m00);
        pts.push_back(Point2f(p));
        width = 2 * cpt[i].x;
        height = 2 * cpt[i].y;
        moveX = cpt[i].x - pts[i].x;
        moveY = cpt[i].y - pts[i].y;
        copyMakeBorder(imgs, imgs, height, height, width, width, BORDER_REPLICATE);
        Rect roi(width - moveX - 0.1 * width, height - moveY - 0.1 * height, 1.2 * width, 1.2 * height);
        //Rect roi(width  - 0.1 * width, height  - 0.1 * height, 1.2 * width, 1.2 * height);
        imgs = imgs(roi);
        width *= 1.4;
        height *= 1.4;
        if (height > width) {
            copyMakeBorder(imgs, imgs, 0.1 * height, 0.1 * height, (height - width) / 2 + 0.1 * height, (height - width) / 2 + 0.1 * height, BORDER_REPLICATE);
        }
        else {
            copyMakeBorder(imgs, imgs, (width - height) / 2 + 0.1 * width, (width - height) / 2 + 0.1 * width, 0.1 * width, 0.1 * width, BORDER_REPLICATE);
        }
        resize(imgs, imgs, Size(1.1 * height, 1.1 * height), 2);
            
        makeFeatureValue(imgs, i);
        title = "Digital no.";
        //showFrame(imgs, 1.1 * height, 1.1 * height);
        imshow(title.append(to_string(count+1)), imgs);

        count++;
    }
}

void resizeTem() {
	int count = 0;
    Mat imgs;
    Moments m;
    int moveX;
    int moveY;
    int width;
    int height;
    for (int i = 0 ; i < nums.size(); i++) {
        imgs = nums[i];
        
        threshold(imgs, imgs, 128, 1, THRESH_BINARY);
        thin th;
        imgs = th.thinImage(imgs, -1);
        imgs = imgs * 255;
        m = moments(imgs, true);
        
        Point p(m.m10 / m.m00, m.m01 / m.m00);
        pts.push_back(Point2f(p));
        width = 2 * cpt[i].x;
        height = 2 * cpt[i].y;
        moveX = cpt[i].x - pts[i].x;
        moveY = cpt[i].y - pts[i].y;
        copyMakeBorder(imgs, imgs, height, height, width, width, BORDER_REPLICATE);
        Rect roi(width - moveX-0.1*width, height - moveY-0.1*height, 1.2*width, 1.2*height);
        imgs = imgs(roi);
        width *= 1.4;
        height *= 1.4;
        if (height > width) {
            copyMakeBorder(imgs, imgs, 0.1*height, 0.1 * height, (height - width) / 2 + 0.1 * height, (height - width) / 2 + 0.1 * height, BORDER_REPLICATE);
        }else {
            copyMakeBorder(imgs, imgs, (width - height) / 2 + 0.1*width, (width - height) / 2 + 0.1 * width, 0.1*width, 0.1*width, BORDER_REPLICATE);
        }
        resize(imgs, imgs, Size(1.1*height, 1.1 * height), 2);

        //fnums.push_back(imgs);
       
        makeTemFeatureValue(imgs, i);

        imshow(to_string(count), imgs);
		
		count++;
	}
}

void saveFontName(string name) {
    ofstream OutFile("fontTemplatePic/fontNames.txt", ios::app);
    OutFile << name << "\n";
    OutFile.close();    
}

string getDir(string filename) {
    string picDir = "E:/cppProject/opencvDemo/opencvDemo01/opencvDemo01/fontTemplatePic/";
    String relDir = picDir.append(filename);
    string finalDir = relDir.append(".png");
    return finalDir;
}

void makeTemplate(string source) {
    //saveFontName(source);
    string finalDir = getDir(source);
    imgTem = getImg(finalDir);
    imgTemP = preprocessingT(imgTem);
    getContours(imgTemP);
    resizeTem();
}

void match(string source) {
    string finalDir = getDir(source);
    
    imgTem = getImg(finalDir);
    cout << finalDir << endl;
    //imshow("source pic", imgTem);
    imgTemP = preprocessingT(imgTem);
    getContours(imgTemP);
    resizePic();
}

int main() {
    match("test");
    //makeTemplate("test");
	waitKey(0);
    
    return 0;
}