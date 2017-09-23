#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <dirent.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

int machine_sample_count;

string file_index;
cv::Mat org;

HOGDescriptor initialize_descriptor_by_file(ifstream &fin){

    HOGDescriptor myHOG(Size(112,24),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，设置HOGDescriptor的检测子,用来计算HOG描述子的

    float val = 0.0f;
    vector<float> myDetector;
    while(!fin.eof())
    {
        fin>>val;
        myDetector.push_back(val);
    }
    fin.close();
    myDetector.pop_back();

    myHOG.setSVMDetector(myDetector);

    return myHOG;
}

void getFiles( string path, vector<string>& files )
{
    DIR  *dir;
    struct dirent  *ptr;
    dir = opendir(path.c_str());
    string pathName;

    while((ptr = readdir(dir)) != NULL){
        if(ptr->d_name[0]!='.'&&ptr->d_name[strlen(ptr->d_name)-4]=='.'){
            files.push_back(pathName.assign(path).append("/").append(string(ptr->d_name)));
        }
    }
}

string get_file_index(string file_name){
    int pre_index=file_name.rfind("_"),post_index=file_name.rfind(".");
    return file_name.substr(pre_index+1,post_index-pre_index-1);
}

Mat HogDetectMulti(Mat &src,HOGDescriptor myHOG){
    Mat drawed_img;
    src.copyTo(drawed_img);

    vector<Rect> found;// vector array of foundlocation

    myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(0,0), 1.15, 2); //1.05->36 levels, 1.1->17 levels, 1.12->14 levels

    //go through all of the detected targets, get the hard examples
    for(int i=0; i < found.size(); i++)
    {
        //resize the window inside the image, because sometime the window out of the image
        Rect r = found[i];

        if(r.x < 0)
            r.x = 0;
        if(r.y < 0)
            r.y = 0;
        if(r.x + r.width > src.cols)
            r.width = src.cols - r.x;
        if(r.y + r.height > src.rows)
            r.height = src.rows - r.y;

        rectangle(drawed_img, r.tl(), r.br(), Scalar(0,255,255), 1);
        machine_sample_count++;
    }
    return drawed_img;
}

int main()
{
    ifstream fin("/Users/lan/Desktop/Papers/FirstConf/experiments/training/SVM_HOG_boot.txt", ios::in);
    HOGDescriptor myHOG=initialize_descriptor_by_file(fin);

    machine_sample_count=0;
    struct timeval timeStart, timeEnd;
    double timeDiff;
    vector<string> file_names;
    //49_0502_800_600/30_0503_800_600/148_0504pm/
    getFiles("/Users/lan/Desktop/TarReg/svm/crop_samples/tobecroped/49_0502_800_600",file_names);

    namedWindow("img");// define a image window
    for(int i=0;i<file_names.size();i++){
        cout<<file_names[i]<<endl;

        org = imread(file_names[i]);
        file_index.assign(get_file_index(file_names[i]));
        gettimeofday(&timeStart,NULL);
        imshow("Detected",HogDetectMulti(org,myHOG));
        gettimeofday(&timeEnd,NULL);
        timeDiff = 1000*(timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_usec - timeStart.tv_usec)/1000; //tv_sec: value of second, tv_usec: value of microsecond
        cout << "Time for one frame : " << timeDiff << " ms" << endl;

        while(true){
            int k=cvWaitKey(10);

            if(k=='d'){
                break;
            }
            if(k=='a'){
                i=(i-2)<0?-1:(i-2);
                break;
            }
        }
    }

    return 0;
}