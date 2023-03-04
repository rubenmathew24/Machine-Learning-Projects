#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


//--------------------------------------------------------------
//Constants
const int MAX_LEN = 2000;
const string FILE_NAME = "titanic.csv";
const int TRAINING_SIZE = 800;



void logisticRegression(vector<bool> x, vector<bool> y, float &w0, float &w1){
	float p_maleSurvival = 0.0;
	float p_femaleSurvival = 0.0;

	int maleAmt = 0, femaleAmt = 0;

	w0 = 0.0;
	w1 = 0.0;

	//calculate p_maleSurvival
	for(int i = 0; i < TRAINING_SIZE; i++){
		if(y[i]){
			p_maleSurvival += (int) x[i];
			maleAmt++;
		}
	}
	p_maleSurvival /= maleAmt;
	//cout << "p_maleSurvival: " << p_maleSurvival << endl;

	float maleOdds = p_maleSurvival/(1-p_maleSurvival);
	//cout << "maleOdds: " << maleOdds << endl;

	//calculate p_femaleSurvival
	for(int i = 0; i < TRAINING_SIZE; i++){
		if(!y[i]){
			p_femaleSurvival += (int) x[i];
			femaleAmt++;
		}
	}
	p_femaleSurvival /= femaleAmt;
	//cout << "p_femaleSurvival: " << p_femaleSurvival << endl;

	float femaleOdds = p_femaleSurvival/(1-p_femaleSurvival);
	//cout << "femaleOdds: " << femaleOdds << endl;

	w0 = log(femaleOdds);
	w1 = log(maleOdds/femaleOdds);


	cout << "\n-------Logistic Regression-------" << endl;
	cout << "Intercept: " << w0 << endl;
	cout << "Slope: " << w1 << endl;
}

void predict(vector<bool> x, vector<bool> y, float w0, float w1){
	int falsePos = 0;
	int falseNeg = 0;
	int truePos = 0;
	int trueNeg = 0;

	for(int i = TRAINING_SIZE; i < x.size(); i++){
		float p = 1/(1+exp(-(w0 + w1*x[i])));
		if(p > 0.5){
			if(y[i]){
				truePos++;
			}
			else{
				falsePos++;
			}
		}
		else{
			if(y[i]){
				falseNeg++;
			}
			else{
				trueNeg++;
			}
		}
	}

	int correct = truePos + trueNeg;
	int incorrect = falsePos + falseNeg;

	cout << "Accuracy: " << (float) (correct)/(correct+incorrect) << endl;
	cout << "Sensitivity: " << (float) (truePos)/(truePos+falseNeg) << endl;
	cout << "Specificity: " << (float) (trueNeg)/(trueNeg+falsePos) << endl;

	cout << "\n-------Confusion Matrix-------" << endl;
	cout << "\t\t\t\t\tPredicted Survived\tPredicted Dead" << endl;
	cout << "Actually Survived\t" << truePos << "\t\t\t\t\t" << falseNeg << endl;
	cout << "Actually Died\t\t" << falsePos << "\t\t\t\t\t" << trueNeg << endl;
}

int main(int argc, char** argv){
	ifstream inFS;	// Input file stream
	string line;
	string _in, pclass_in, survived_in, sex_in, age_in;
	vector<double> something(MAX_LEN);
	vector<int> pclass(MAX_LEN);
	vector<bool> survived(MAX_LEN);
	vector<bool> sex(MAX_LEN);
	vector<double> age(MAX_LEN);

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

		getline(inFS, _in, ',');
		getline(inFS, pclass_in, ',');
		getline(inFS, survived_in, ',');
		getline(inFS, sex_in, ',');
		getline(inFS, age_in, '\n');

		//something.at(numObservations) = stof(_in);
		pclass.at(numObservations) = stoi(pclass_in);
		survived.at(numObservations) = (bool) stoi(survived_in);
		sex.at(numObservations) = (bool) stoi(sex_in);
		age.at(numObservations) = stof(age_in);

		numObservations++;
	}

	//something.resize(numObservations);
	pclass.resize(numObservations);
	survived.resize(numObservations);
	sex.resize(numObservations);
	age.resize(numObservations);

	cout << "new length: " << something.size() << endl;

	// Close file
	cout << "Closing file " << FILE_NAME << endl;
	inFS.close(); // Done with file, so close it

	cout << "Number of records: " << numObservations << endl;


	float w0, w1;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	logisticRegression(survived, sex, w0, w1);
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	predict(survived, sex, w0, w1);

	cout << "\n(Training took " << duration_cast<microseconds>(stop - start).count() << " microseconds)" << endl;
	cout << "Program terminated." << endl;

	return 0;
}