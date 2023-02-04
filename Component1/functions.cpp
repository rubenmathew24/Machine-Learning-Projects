#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;


//--------------------------------------------------------------
//Constants
const int MAX_LEN = 1000;
const string FILE_NAME = "Boston.csv";


//--------------------------------------------------------------
//Single Variable Statistics

double sum(vector<double> x){
	double sum = 0;
	for (int i = 0; i < x.size(); i++){
		sum += x.at(i);
	}
	return sum;
}

double mean(vector<double> x){
	return sum(x) / x.size();
}

double median(vector<double> x){
	sort(x.begin(), x.end());
	int middle = x.size()/2;

	if (x.size() % 2 == 0) return (x.at(middle) + x.at(middle-1)) / 2;
	
	return x.at(middle);
}

vector<double> range(vector<double> x){
	vector<double> range(2);
	sort(x.begin(), x.end());
	range.at(0) = x.at(0);
	range.at(1) = x.at(x.size()-1);
	return range;
}

//--------------------------------------------------------------
//Two Variable Statistics
double covar(vector<double> x, vector<double> y){
	vector<double> xy(MAX_LEN);
	double x_mean = mean(x);
	double y_mean = mean(y);
	int n = x.size();

	for (int i = 0; i < n; i++){
		xy.at(i) = (x.at(i) - x_mean) * (y.at(i) - y_mean);
	}

	return sum(xy) / (n-1);
}

double cor(vector<double> x, vector<double> y){
	return covar(x, y) / (sqrt(covar(x,x)) * sqrt(covar(y,y)));
}


//--------------------------------------------------------------
//Print Statistics
void print_stats(vector<double> x){
	cout << "Sum = " << sum(x) << endl;
	cout << "Mean = " << mean(x) << endl;
	cout << "Median = " << median(x) << endl;

	vector<double> r = range(x);
	cout << "Range = [" << r.at(0) << ", " << r.at(1) << "]" << endl;
}

int main(int argc, char** argv){
	ifstream inFS;	// Input file stream
	string line;
	string rm_in, medv_in;
	vector<double> rm(MAX_LEN);
	vector<double> medv(MAX_LEN);

	// Try to open file
	cout << "Opening file " << FILE_NAME << endl;

	inFS.open(FILE_NAME);
	if (!inFS.is_open()) {
		cout << "Could not open file " << FILE_NAME << endl;
		return 1; // 1 indicates error
	}

	// Can now use inFS stream like cin stream
	// File should contain 2 doubles

	cout << "Reading line 1" << endl;
	getline(inFS, line);

	//echo heading
	cout << "heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good()) {

		getline(inFS, rm_in, ',');
		getline(inFS, medv_in, '\n');

		rm.at(numObservations) = stof(rm_in);
		medv.at(numObservations) = stof(medv_in);

		numObservations++;
	}

	rm.resize(numObservations);
	medv.resize(numObservations);

	cout << "new length: " << rm.size() << endl;

	// Close file
	cout << "Closing file " << FILE_NAME << endl;
	inFS.close(); // Done with file, so close it

	cout << "Number of records: " << numObservations << endl;

	cout << "\nStats for rm" << endl;
	print_stats(rm);

	cout << "\nStats for medv" << endl;
	print_stats(medv);

	cout << "\n Covariance = " << covar(rm, medv) << endl;
	cout << "\n Correlation = " << cor(rm, medv) << endl;

	cout << "\nProgram terminated." << endl;

	return 0;
}