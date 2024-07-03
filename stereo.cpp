#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {

	// Check if the correct number of arguments are provided
	if (argc != 4) {
		std::cerr << "usage: <calibration_file> <left_image> <right_image>" << std::endl;
		return 1;
	}

	// Retrieve the arguments
	const char* calibrationFile1 = argv[1];
	const char* imageFile1 = argv[2];
	const char* imageFile2 = argv[3];

	// Read the arguments back
	std::cout << "Calibration File: " << calibrationFile1 << std::endl;
	std::cout << "Left Image File: " << imageFile1 << std::endl;
	std::cout << "Right Image File: " << imageFile2 << std::endl;

	//claibration file is generated from a calibration of a two camera stereo pair
	//std::string calibrationFile1 = "C:\\Users\\rolan\\OneDrive\\Documents\\DickinsonWork\\2023-2024\\COMP342\\stereo\\stereo2024\\calibration.json"; // This will depend on where you saved it
	//std::string imageFile1 = "C:\\Users\\rolan\\OneDrive\\Documents\\DickinsonWork\\2023-2024\\COMP342\\stereo\\StereoPairs\\bell_left.jpg";
	//std::string imageFile2 = "C:\\Users\\rolan\\OneDrive\\Documents\\DickinsonWork\\2023-2024\\COMP342\\stereo\\StereoPairs\\bell_right.jpg";

	//some variables needed for stereo rectify
	cv::Mat K1, K2, R, T;
	std::vector<double> d1, d2;
	cv::Mat distCoeffsL, distCoeffsR;
	cv::Size imageSize;
	
	//ensure the calibration file is readable
	cv::FileStorage fs(calibrationFile1, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cerr << "Failed to open calibration.json" << std::endl;
		return -1;
	}

	//populating variables with calibrationFile1 data
	cv::Mat cameraMatrixL, cameraMatrixR;
	fs["CameraMatrixL"] >> K1;
	fs["cameraMatrixR"] >> K2;
	fs["R"] >> R;
	fs["T"] >> T;
	fs["imageSize"] >> imageSize;
	fs["distCoeffsL"] >> distCoeffsL;
	fs["distCoeffsR"] >> distCoeffsR;
	
	//convert distCoeffs to doubles and assign to d1 and d1 variables
	d1.assign(distCoeffsL.begin<double>(), distCoeffsL.end<double>());
	d2.assign(distCoeffsR.begin<double>(), distCoeffsR.end<double>());

	//outputs for the stereo rectitify function
	cv::Mat R1;
	cv::Mat R2;
	cv::Mat P1;
	cv::Mat P2;
	cv::Mat Q;

	//Computing the stereo rectification
	cv::stereoRectify(K1, d1, K2, d2, imageSize, R, T, R1, R2, P1, P2, Q);

	//Get the original Images
	cv::Mat I1 = cv::imread(imageFile1);
	cv::Mat I2 = cv::imread(imageFile2);

	//input for undistort map
	float type = CV_32F;

	//outputs for undistort map
	cv::Mat LeftMap1;
	cv::Mat LeftMap2;

	cv::Mat RightMap1;
	cv::Mat RightMap2;

	//create undistort maps
	cv::initUndistortRectifyMap(K1, d1, R1, P1, imageSize, type, LeftMap1, LeftMap2);
	cv::initUndistortRectifyMap(K2, d2, R2, P2, imageSize, type, RightMap1, RightMap2);

	//remap outputs (rectified Images)
	cv::Mat targetImage1;
	cv::Mat targetImage2;

	//remap
	cv::remap(I1, targetImage1, LeftMap1, LeftMap2, cv::INTER_LINEAR);
	cv::remap(I2, targetImage2, RightMap1, RightMap2, cv::INTER_LINEAR);

	//disparity estimation
	//
	//shrink images
	cv::Mat smallImage1;
	cv::resize(targetImage1, smallImage1, targetImage1.size() / 4, 0, 0, cv::INTER_AREA);

	cv::Mat smallImage2;
	cv::resize(targetImage2, smallImage2, targetImage2.size() / 4, 0, 0, cv::INTER_AREA);

	//Block Matching
	//
	//Convert images to greyscale to Block Match an image
	cv::Mat greyscaleImage1;
	cv::cvtColor(smallImage1, greyscaleImage1, cv::COLOR_BGR2GRAY);
	cv::Mat greyscaleImage2;
	cv::cvtColor(smallImage2, greyscaleImage2, cv::COLOR_BGR2GRAY);

	//place to store a value from the Match Compute method
	cv::Mat disparity;
	cv::Mat disparity1;

	//created to hold the images generated from the matching function
	std::vector<cv::Mat> Images;

	//user Message
	std::cout << "\n\n\n" <<
		"This program requires no block size or Max disparity as its puropse is to quickly determine the best settings. \n" <<
		"This is achieved by computing a disparity map of all block sizes from 5-43 and Max disparites 16-256 for a single picture" <<
		"The process may take a while... \n" <<
		"You may navigate the results using the arrowkeys"
		<< std::endl;

	//variables needed for the upcoming loop operations
	cv::Mat concatenated_image;
	int row = 0;

	//variables for semi-global matching
	int matchingP1 = 100;
	int matchingP2 = 1000;
	int disp12MaxDiff = 1;
	int preFilterCap = 0;
	int uniquenessRatio = 5;
	int speckleWindowSize = 400;
	int speckleRange = 200;

	//first loop iterates through blockSize
	for (int blockSize = 11; blockSize <= 23; blockSize = blockSize + 2) {

		//variables that are created for the loop operations
		std::vector<cv::Mat> ImageGrids(4);
		cv::Mat display;
		int row = 0;

		//second loops iterates over maxDisparity
		for (int maxDisparity = 16; maxDisparity <= 256; maxDisparity = maxDisparity + 16) {

			//Block Matching
			cv::Ptr<cv::StereoBM> blockMatcher = cv::StereoBM::create((float)maxDisparity, (float)blockSize);
			blockMatcher->compute(greyscaleImage1, greyscaleImage2, disparity);

			//SemiGlobal matching
			/*cv::Ptr<cv::StereoSGBM> sgMatcher = cv::StereoSGBM::create(0, maxDisparity, blockSize, matchingP1, matchingP2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange);
			sgMatcher->compute(greyscaleImage1, greyscaleImage2, disparity);*/

			//converting disparity to an 8 bit image
			disparity.convertTo(display, CV_8UC1, 255.0 / (16 * maxDisparity));

			//populate the first ImageGrids row if empty
			//this is a strange way of indicating the first row, but when I got it working I stopped
			if (ImageGrids[row].empty()) {
				ImageGrids[row] = display;
			}
			//populate the following image gridrow with the current disparity map concatenated to the previous ones
			//basically this is lining four images up in a row next to one another
			else {
				hconcat(ImageGrids[row], display, concatenated_image);

				ImageGrids[row] = concatenated_image;
			}

			//When the first row of ImageGrids is populated, move to the next row
			//this is janky, but when I got it working I stopped
			if (maxDisparity % 64 == 0) {
				row = row + 1;
			}
		}

		//ImageGridFull holds the 
		cv::Mat ImageGridFull = ImageGrids[0];

		//this takes the images rows that we created and stacks them into a 4x4 grid
		for (int i = 1; i < 4; ++i) {
			vconcat(ImageGridFull, ImageGrids[i], concatenated_image);

			ImageGridFull = concatenated_image;
		}

		//resize the image so the image grids fit on a standard screen size
		cv::resize(ImageGridFull, ImageGridFull, cv::Size(), .75, .75, cv::INTER_AREA);

		//add each grid to the Images Vector
		Images.push_back(ImageGridFull);

		//prepare the variables for the next iteration
		ImageGrids.clear();
		display.release();
		row = 0;

		//tell the user how many trials till complete
		std::cout << "Computed " << (blockSize - 9) / 2 * 16 << " of 112 disparity maps" << std::endl;
	}

	//display the images
	cv::namedWindow("Window");
	cv::imshow("Window", Images[0]);
	cv::waitKeyEx();

	//display the images and allow the user to browse the image grids them with the arrow keys
	int i = 0;
	while (true) {
		cv::namedWindow("Window");
		cv::imshow("Window", Images[i]);
		std::cout << "Showing disparity maps with block sizes: " << i*2+11 << "| Press esc to exit" << std::endl;

		int key = cv::waitKeyEx();
		
		if (key == 2424832 && i > 0) {
			i = i - 1;
		} 
		else if (key == 2555904 && i < Images.size()-1) {
			i = i + 1;
		}
		else if (key == 27) {
			break;
		}
	}

	return 0;

}
