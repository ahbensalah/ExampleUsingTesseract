//Qt
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDir>

//C++
#include <memory>
#include <vector>
#include <string>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <string.h>
#include <math.h>

//Tesseract
#include <allheaders.h>
#include <baseapi.h>

//OpenCV
//#include <opencv/cv.h>
//#include <opencv/ml.h>

#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat backgroundDataPreparation(vector<string> paths, int sampleSize) {

    /*
     * This function prepare the sample that we used to sperate id cart pixels from background pixel
     */

    Mat backSample = Mat(sampleSize, 3, CV_32FC1);

    vector<Mat> background_samples;
    for(string path : paths)    {
        Mat imgBack = imread(path, IMREAD_COLOR);
        int indice =0;
        Mat training_back = Mat(imgBack.rows*imgBack.cols, 3, CV_32FC1);
        for(int i= 0; i<imgBack.rows; ++i){
            for(int j=0; j < imgBack.cols; ++j){
                if(indice < training_back.rows)    {
                    Vec3b bgrPixel = imgBack.at<Vec3b>(i, j);
                    training_back.at<float>(indice, 0) = bgrPixel.val[0];
                    training_back.at<float>(indice, 1) = bgrPixel.val[1];
                    training_back.at<float>(indice, 2) = bgrPixel.val[2];
                    indice++;
                }
            }
        }
        background_samples.push_back(training_back);
    }

    int indice =0;
    while(indice < sampleSize){
        srand (time(NULL));
        int image_choice_index = rand()%background_samples.size();
        Mat training_back = background_samples.at(image_choice_index);
        srand (time(NULL));
        int training_row_data = rand()%training_back.rows;

        backSample.at<float>(indice, 0) = training_back.at<float>(training_row_data,0);
        backSample.at<float>(indice, 1) = training_back.at<float>(training_row_data,1);
        backSample.at<float>(indice, 2) = training_back.at<float>(training_row_data,2);
        indice++;
    }
    return backSample;
}

Mat idCarteDetection(Mat img_testing_data, string training_idImage_path)  {
    /*
     * This function use a naive bayesienne classifier to detect the id cart pixels
     */
    CvNormalBayesClassifier classifier;

    //Mat idImg = imread("C:/Users/Ahmed/Documents/TestTechnique/dataset/04_aut_id/images/04_aut_id.jpg", IMREAD_COLOR);
    Mat idImg = imread(training_idImage_path, IMREAD_COLOR);
    Mat training_data = Mat(2*(idImg.rows*idImg.cols), 3, CV_32FC1);
    Mat response_training_data = Mat::zeros(training_data.rows,1, CV_32FC1);
    int indice = 0;
    for(int i= 0; i<idImg.rows; ++i){
        for(int j=0; j < idImg.cols; ++j){
            if(indice < (idImg.rows*idImg.cols))    {
                Vec3b bgrPixel = idImg.at<Vec3b>(i, j);
                training_data.at<float>(indice, 0) = bgrPixel.val[0];
                training_data.at<float>(indice, 1) = bgrPixel.val[1];
                training_data.at<float>(indice, 2) = bgrPixel.val[2];
                response_training_data.at<float>(indice,0) = 1.0;
                indice++;
            }
        }
    }

    Mat imgBack = imread("C:/Users/Ahmed/Documents/TestTechnique/Projet/OCRExample/data/back4.png", IMREAD_COLOR);
    Mat training_back = Mat(imgBack.rows*imgBack.cols, 3, CV_32FC1);
    for(int i= 0; i<imgBack.rows; ++i){
        for(int j=0; j < imgBack.cols; ++j){
            if(indice < training_data.rows)    {
                Vec3b bgrPixel = imgBack.at<Vec3b>(i, j);
                training_data.at<float>(indice, 0) = bgrPixel.val[0];
                training_data.at<float>(indice, 1) = bgrPixel.val[1];
                training_data.at<float>(indice, 2) = bgrPixel.val[2];
                response_training_data.at<float>(indice,0) = 0.0;
                indice++;
            }
        }
    }

    classifier.train(training_data,response_training_data, Mat(), Mat(), false);

    Mat testing_data = Mat(img_testing_data.rows*img_testing_data.cols, 3, CV_32FC1);
    Mat results = Mat(img_testing_data.rows*img_testing_data.cols, 1, CV_32FC1);
    int test_indice =0;

    for(int i= 0; i<img_testing_data.rows; ++i){
        for(int j=0; j < img_testing_data.cols; ++j){
            if(test_indice < testing_data.rows)    {
                Vec3b bgrPixel = img_testing_data.at<Vec3b>(i, j);
                testing_data.at<float>(test_indice, 0) = bgrPixel.val[0];
                testing_data.at<float>(test_indice, 1) = bgrPixel.val[1];
                testing_data.at<float>(test_indice, 2) = bgrPixel.val[2];
                test_indice++;
            }
        }
    }

    classifier.predict(testing_data, &results);
    Mat imgResults = Mat::zeros(img_testing_data.rows, img_testing_data.cols, CV_32FC1);
    test_indice =0;
    for(int i=0; i < img_testing_data.rows; ++i){
        for(int j=0; j < img_testing_data.cols; ++j){
            imgResults.at<float>(i,j) = results.at<float>(test_indice,0);
            test_indice++;
        }
    }

    // Create a structuring element (SE)
    int morph_size = 2;
    Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    Mat masque; // result matrix


    morphologyEx( imgResults, masque, MORPH_OPEN, element, Point(-1,-1),3);
    morphologyEx( masque, masque, MORPH_CLOSE, element, Point(-1,-1),3);

    //imwrite("C:/Users/Ahmed/Documents/TestTechnique/dataset/04_aut_id/images/CA/classifRes0.png", masque);

    return masque;
}

float compute_skew2(Mat input){
    cv::Mat gray;
    cv::cvtColor(input,gray,CV_BGR2GRAY);

    // since your image has compression artifacts, we have to threshold the image
    int threshold = 100;
    cv::Mat mask = gray > threshold;
    // extract contours
    float  angle = 0.0;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    qDebug() << "compute_skew2 contours.size = " << contours.size();
    for(int i=0; i<contours.size(); ++i)
    {
        // fit bounding rectangle around contour
        cv::RotatedRect rotatedRect = cv::minAreaRect(contours[i]);
        // read points and angle
        cv::Point2f rect_points[4];
        rotatedRect.points( rect_points );
        angle = rotatedRect.angle; // angle
        qDebug() << "compute_skew2 angle: " << angle;
    }
    qDebug() << "END";
    return angle;
}

double compute_skew(Mat input){
    double angle = 0;
    cv::Size size = input.size();
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(input, lines, 1, CV_PI/180, 100, size.width / 2.f, 20);
    cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));
    unsigned nb_lines = lines.size();
    for (unsigned i = 0; i < nb_lines; ++i)
    {
        cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
                 cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0 ,0));
        angle += atan2((double)lines[i][3] - lines[i][1],
                       (double)lines[i][2] - lines[i][0]);
    }
    angle /= nb_lines; // mean angle, in radians.
    qDebug() << "angle: " << angle * 180 / CV_PI;

    cv::imshow("compute_skew", disp_lines);
    cv::waitKey(0);
    cv::destroyWindow("compute_skew");
    return angle;
}

/**
 * @brief deskew
 * @param img
 * @param angle
 * @param blobs
 */
void deskew(Mat img, double angle, std::vector < std::vector<cv::Point2i> > blobs, string path)   {
    /*
     * This function correct the skewness of idBlob
     */
    vector<Mat> imageList;
    for(size_t i=0; i < blobs.size(); i++) {
        cv::Mat blobImage = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::Mat output = img.clone();
        unsigned char val = 1.0;
        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;

            blobImage.at<uchar>(y,x) = val;
        }

        std::vector<cv::Point> points;
        cv::Mat_<uchar>::iterator it = blobImage.begin<uchar>();
        cv::Mat_<uchar>::iterator end = blobImage.end<uchar>();

        for (; it != end; ++it)
            if (*it)
              points.push_back(it.pos());

        cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));        
        //angle = compute_skew(blobImage);
        //compute_skew2(blobImage);
        /*
        qDebug() << "****======>" << box.angle
                 << "box.size.width = " << box.size.width
                 << "box.size.height = " << box.size.height;
        */
        angle = box.angle;
        if (box.size.width < box.size.height) {
          angle = angle - 90;
        }
        cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, angle, 1);
/*
        Point2f vertices[4];
        box.points(vertices);

        Mat contourImage= Mat::zeros(img.size(), CV_8UC3);
        Point2f secondDiagPoint;
        double distance = 0.0;
        for(unsigned int j=1; j<4; ++j){
            double p_distance = std::sqrt(std::pow((vertices[0].x - vertices[j].x),2.0) + std::pow((vertices[0].y - vertices[j].y),2.0));
            qDebug() << p_distance << "<?>" << distance;
            if(distance < p_distance){
                secondDiagPoint = vertices[j];
                distance = p_distance;
            }
        }

        for(unsigned int j=0; j<4; ++j){
            cv::rectangle(contourImage, vertices[0], secondDiagPoint, cv::Scalar(255,255,255), -1);
            //cv::line(contourImage, vertices[j], vertices[(j+1)%4], cv::Scalar(255,255,255),50);
        }

        imwrite(path+"/contour_"+QString::number(i).toStdString()+".png", contourImage);
        cv::Mat gray;
        cv::cvtColor(contourImage,gray,CV_BGR2GRAY);

        // since your image has compression artifacts, we have to threshold the image
        int threshold = 100;
        cv::Mat mask = gray > threshold;
*/

        for(int row = box.boundingRect().y; row < (box.boundingRect().y + box.boundingRect().height); ++row){
            for(int col = box.boundingRect().x; col < (box.boundingRect().x + box.boundingRect().width); ++col){
                if(row > 0 && col > 0 && row < blobImage.rows && col < blobImage.cols)
                    blobImage.at<uchar>(row,col) = 1.0;
            }
        }


        for(int row = 0; row < blobImage.rows; ++row){
            for(int col = 0; col < blobImage.cols; ++col){
                Vec3b bgrPixel = output.at<Vec3b>(row, col);
                output.at<Vec3b>(row, col)[0] = bgrPixel.val[0] * blobImage.at<uchar>(row,col);
                output.at<Vec3b>(row, col)[1] = bgrPixel.val[1] * blobImage.at<uchar>(row,col);
                output.at<Vec3b>(row, col)[2] = bgrPixel.val[2] * blobImage.at<uchar>(row,col);
            }
        }
        //imwrite(path+"/Output_"+QString::number(i).toStdString()+".png", output);

        cv::Mat rotated;
        cv::warpAffine(output, rotated, rot_mat, output.size(), cv::INTER_CUBIC);
        imageList.push_back(rotated);

        cv::Size box_size = box.size;
        if (angle < -45.)
            std::swap(box_size.width, box_size.height);
        cv::Mat cropped;
        cv::getRectSubPix(rotated, box_size, box.center, cropped);

        //imwrite(path+"/CRotated_"+QString::number(i).toStdString()+".png", rotated);
        imwrite(path+"/rotated_"+QString::number(i).toStdString()+".png", cropped);
    }
}

void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);

            std::vector <cv::Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }

    // Randomy color the blobs
    /*
    cv::Mat output = cv::Mat::zeros(binary.size(), CV_8UC3);
    for(size_t i=0; i < blobs.size(); i++) {
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));

        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;

            output.at<cv::Vec3b>(y,x)[0] = b;
            output.at<cv::Vec3b>(y,x)[1] = g;
            output.at<cv::Vec3b>(y,x)[2] = r;
        }
    }
    cv::imshow("labelled", output);
    imwrite("C:/Users/Ahmed/Documents/TestTechnique/dataset/04_aut_id/images/CA/classifRes0.png", output);
    */

}

void writeJson(vector<string> words, vector<double> confidances, vector<vector<int>> coordinates, double recognitionRate, QString path, QString name)  {
    /*
     * This function export tesseract results in json file
     */
    QFile file(path + name);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return;

    qDebug() << "exportResults" << words.size() << ", " << confidances.size();
    QTextStream out(&file);
    out << "{ \n";
    out << "\"recognitionRate\": \"" << QString::number(recognitionRate) <<"\", \n";
    for(int i=0; i < words.size(); ++i){        
        out << "\"field" << QString::number(i) << "\": {\n";
        out << "\"quad\":" << "[[" << QString::number(coordinates[i].at(0)) << ", " << QString::number(coordinates[i].at(1)) << "], "
            << "[" << QString::number(coordinates[i].at(2)) << ", " << QString::number(coordinates[i].at(3)) << "]], \n";
        out << "\"value\": \"" << QString::fromStdString(words[i]) << "\"\n";
        out << "},\n";
    }
    out << "}";
    file.close();
}

vector<string> readJson(string groundTruthFilePath) {
    QFile file;
    QString val;
    file.setFileName(QString::fromStdString(groundTruthFilePath));
    file.open(QIODevice::ReadOnly | QIODevice::Text);
    val = file.readAll();
    file.close();

    vector<string> true_words;

    QJsonDocument jsonResponse = QJsonDocument::fromJson(val.toUtf8());
    QJsonObject jsonObject = jsonResponse.object();
    //convert the json object to variantmap
    QVariantMap mainMap = jsonObject.toVariantMap();

    for(QString key : mainMap.keys()){
        QVariantMap mapContent = mainMap[key].toMap();
        if(key.contains("field")){
            true_words.push_back(mapContent["value"].toString().toStdString());
        }
    }
    return true_words;
}

double verificationWithGroundTruth(string groundTruthFilePath, vector<string> words){
    vector<string> ground_truth = readJson(groundTruthFilePath);
    double founded_words_nb = 0.0;
    for(string word_ground_truth : ground_truth){
        for(string word_ocr : words){
            if(word_ground_truth.compare(word_ocr) == 0){
                founded_words_nb++;
                break;
            }
        }
    }
    return founded_words_nb/static_cast<double>(ground_truth.size());
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    vector<string> back_path_images = {"C:/Users/Ahmed/Documents/TestTechnique/Projet/OCRExample/data/back1.png",
                                      "C:/Users/Ahmed/Documents/TestTechnique/Projet/OCRExample/data/back2.png",
                                      "C:/Users/Ahmed/Documents/TestTechnique/Projet/OCRExample/data/back3.png",
                                      "C:/Users/Ahmed/Documents/TestTechnique/Projet/OCRExample/data/back4.png"};

    QString rootPath = "C:/Users/Ahmed/Documents/TestTechnique/dataset";
    QDir rootDirectory(rootPath);
    QStringList rootDirecotryContent = rootDirectory.entryList();
    for(QString rootContentName : rootDirecotryContent){
        if(rootContentName != "." && rootContentName != ".."){
            qDebug() << "======>>>" << rootContentName.split("_");
            QString used_language = rootContentName.split("_")[1];
            QString ground_truth_path;
            QString training_idImage_path;
            vector<QString> test_image_paths;

            QString subDirectoryPath =  rootPath + "/" + rootContentName;
            QDir subDirectory(subDirectoryPath);
            QStringList subDirecotryContent = subDirectory.entryList();
            for(QString subDirecotryName : subDirecotryContent){
                if(subDirecotryName != "." && subDirecotryName != ".."){
                    if(subDirecotryName.contains(".json")){
                        ground_truth_path = subDirectoryPath + "/" + subDirecotryName;
                    } else if (subDirecotryName.contains("images")){
                        QString imagesDirectoryPath =  subDirectoryPath + "/" + subDirecotryName;
                        QDir imagesDirectory(imagesDirectoryPath);
                        QStringList imagesDirecotryContentList = imagesDirectory.entryList();
                        for(QString imagesDirecotryItem : imagesDirecotryContentList){
                            if(imagesDirecotryItem!="." && imagesDirecotryItem!=".."){
                                QFileInfo fileInfo(imagesDirectoryPath+"/"+imagesDirecotryItem);
                                qDebug() << imagesDirecotryItem;
                                if(fileInfo.isFile() && imagesDirecotryItem.contains(".jpg")){
                                    training_idImage_path = fileInfo.absoluteFilePath();
                                }   else if(fileInfo.isDir()){
                                    QString subImagesDirectoryPath = fileInfo.absoluteFilePath();
                                    QDir subImagesDirectory(subImagesDirectoryPath);
                                    QStringList subImagesDirectoryContentList = subImagesDirectory.entryList();
                                    for (QString testImageName : subImagesDirectoryContentList) {
                                        QFileInfo imageFileInfo(subImagesDirectoryPath+"/"+testImageName);
                                        if(imageFileInfo.isFile() && (testImageName.contains(".jpg") || testImageName.contains(".png"))){
                                            test_image_paths.push_back(imageFileInfo.absoluteFilePath());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            QDir ocrDirectory(rootPath + "/" + rootContentName);
            ocrDirectory.mkpath("OCR/");

            for(QString test_image_path : test_image_paths) {

            //QString test_image_path = "C:/Users/Ahmed/Documents/TestTechnique/dataset/10_cze_id/images/CA/CA10_10.jpg";
            //training_idImage_path = "C:/Users/Ahmed/Documents/TestTechnique/dataset/10_cze_id/images/10_cze_id.jpg";
            //ground_truth_path = "C:/Users/Ahmed/Documents/TestTechnique/dataset/10_cze_id/10_cze_id.json";

                QFileInfo test_image_fileInfo(test_image_path);

                Mat img_testing_data = imread(test_image_path.toStdString(), IMREAD_COLOR);

                Mat masque = idCarteDetection(img_testing_data, training_idImage_path.toStdString());

                std::vector < std::vector<cv::Point2i> > blobs;
                FindBlobs(masque, blobs);

                double angle = 0.0;
                deskew(img_testing_data, angle, blobs, (rootPath + "/" + rootContentName+ "/" + "OCR/").toStdString());

                tesseract::TessBaseAPI tess;
                string appliedLang = "eng";
                if(used_language.toLower() == "aut")
                    appliedLang ="deu";
                else if(used_language.toLower() == "cze")
                    appliedLang ="ces";
                else if(used_language.toLower() == "deu")
                    appliedLang ="deu";
                else if(used_language.toLower() == "esp")
                    appliedLang ="spa";
                else if(used_language.toLower() == "est")
                    appliedLang ="est";
                else if(used_language.toLower() == "fin")
                    appliedLang ="fin";
                else if(used_language.toLower() == "hun")
                    appliedLang ="hun";
                else if(used_language.toLower() == "ita")
                    appliedLang ="ita";
                else if(used_language.toLower() == "mac")
                    appliedLang ="eng";
                else if(used_language.toLower() == "mda")
                    appliedLang ="ron";
                else if(used_language.toLower() == "pol")
                    appliedLang ="pol";
                else if(used_language.toLower() == "prt")
                    appliedLang ="por";
                else if(used_language.toLower() == "srb")
                    appliedLang ="srp";

                if (tess.Init("C:/Users/Ahmed/Documents/TestTechnique/Projet/tessdata", appliedLang.c_str())) {
                    qDebug() << "OCRTesseract: Could not initialize tesseract.";
                }

                // setup
                tess.SetPageSegMode(tesseract::PageSegMode::PSM_AUTO);
                tess.SetVariable("save_best_choices", "T");
                tess.SetVariable("tessedit_write_images", "T");

                vector<string> words;
                vector<double> confidances;
                vector<vector<int>> coordinates;

                for(int i = 0; i<blobs.size(); ++i){
                //int i = 6;
                    // read image
                    string pathImage = rootPath.toStdString() + "/" + rootContentName.toStdString()+ "/" + "OCR/rotated_"+QString::number(i).toStdString()+".png";
                    auto pixs = pixRead(pathImage.c_str());

                    if (!pixs){
                        qDebug() << "Cannot open input file: " << QString::fromStdString(pathImage);
                        continue;
                    }

                    // recognize
                    tess.SetImage(pixs);
                    tess.Recognize(0);

                    tesseract::ResultIterator* ri = tess.GetIterator();
                    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;

                    if (ri != 0) {
                        do {
                          const char* word = ri->GetUTF8Text(level);
                          float conf = ri->Confidence(level);
                          int x1, y1, x2, y2;
                          ri->BoundingBox(level, &x1, &y1, &x2, &y2);
                          if(!QString(word).isEmpty() && QString(word)!= " "){
                            words.push_back(QString(word).toStdString());
                            confidances.push_back(conf);
                             vector<int> coords = {x1, y1, x2, y2};
                            coordinates.push_back(coords);
                          }

                          delete[] word;
                        } while (ri->Next(level));
                    }
                    pixDestroy(&pixs);
                    QFile imageFile(QString::fromStdString(pathImage));
                    imageFile.remove();
                }

                double recognition_rate = verificationWithGroundTruth(ground_truth_path.toStdString(), words);

                QString fileNameJson = "/"+test_image_fileInfo.fileName()+"_OCRExampleResult.json";

                writeJson(words, confidances, coordinates, recognition_rate, (rootPath + "/" + rootContentName+ "/OCR"), fileNameJson);

                tess.Clear();
            }
        }
    }

    /*
    Mat img_testing_data = imread("C:/Users/Ahmed/Documents/TestTechnique/dataset/04_aut_id/images/CA/CA04_30.jpg", IMREAD_COLOR);
    Mat masque = idCarteDetection(img_testing_data);

    std::vector < std::vector<cv::Point2i> > blobs;
    FindBlobs(masque, blobs);

    double angle = 0.0;
    deskew(img_testing_data, angle, blobs);

    tesseract::TessBaseAPI tess;
    if (tess.Init("C:/Users/Ahmed/Documents/TestTechnique/Projet/tessdata", "eng"))
    {
        qDebug() << "OCRTesseract: Could not initialize tesseract.";
    }

    // setup
    tess.SetPageSegMode(tesseract::PageSegMode::PSM_AUTO);
    tess.SetVariable("save_best_choices", "T");
    tess.SetVariable("tessedit_write_images", "T");

    vector<string> words;
    vector<double> confidances;
    vector<vector<int>> coordinates;

    for(int i = 0; i<blobs.size(); ++i){
        // read image
        string pathImage = "C:/Users/Ahmed/Documents/TestTechnique/dataset/04_aut_id/images/CA/rotated"+QString::number(i).toStdString()+".png";
        auto pixs = pixRead(pathImage.c_str());

        if (!pixs){
            qDebug() << "Cannot open input file: " << "C:/Users/Ahmed/Documents/TestTechnique/dataset/04_aut_id/images/04_aut_id.jpg";
            continue;
        }

        // recognize
        tess.SetImage(pixs);
        tess.Recognize(0);

        tesseract::ResultIterator* ri = tess.GetIterator();
        tesseract::PageIteratorLevel level = tesseract::RIL_WORD;

        if (ri != 0) {
            do {
              const char* word = ri->GetUTF8Text(level);
              float conf = ri->Confidence(level);
              int x1, y1, x2, y2;
              ri->BoundingBox(level, &x1, &y1, &x2, &y2);
              if(!QString(word).isEmpty() && QString(word)!= " "){
                words.push_back(QString(word).toStdString());
                confidances.push_back(conf);
                 vector<int> coords = {x1, y1, x2, y2};
                coordinates.push_back(coords);
              }

              delete[] word;
            } while (ri->Next(level));
        }
        pixDestroy(&pixs);
        QFile imageFile(QString::fromStdString(pathImage));
        imageFile.remove();
    }

    double recognition_rate = verificationWithGroundTruth("C:/Users/Ahmed/Documents/TestTechnique/dataset/04_aut_id/04_aut_id.json", words);

    writeJson(words, confidances, coordinates, "C:/Users/Ahmed/Documents/TestTechnique/dataset/04_aut_id");

    tess.Clear();
*/
    return a.exec();
}
