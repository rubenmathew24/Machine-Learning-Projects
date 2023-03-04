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

//--------------------------------------------------------------
//Globals
double p_class_survived[2][3];
double age_survived[2][2];
double sex_survived[2][2];
double apriori_survived;

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

double var(vector<double> x){
	double m = mean(x);
	double sum = 0;
	for (int i = 0; i < x.size(); i++){
		sum += pow(x.at(i) - m, 2);
	}
	return sqrt(sum / (x.size()-1));
}

//pclass, sex, age
float raw_prob(int w, bool y, double z){
	float num_s = p_class_survived[0][w-1] * sex_survived[0][y] * apriori_survived;
	num_s *= 1 / sqrt(2 * M_PI * age_survived[0][1]) * exp(-pow((z-age_survived[0][0]),2)/(2 * age_survived[0][1]));
	float num_p = p_class_survived[1][w-1] * sex_survived[1][y] * (1-apriori_survived);
	num_p *= 1 / sqrt(2 * M_PI * age_survived[1][1]) * exp(-pow((z-age_survived[1][0]),2)/(2 * age_survived[1][1]));

	return num_s / (num_s + num_p);
}

//pclass, survived, sex, age
void naive_bayes(vector<int> w, vector<bool> x, vector<bool> y, vector<double> z){
	cout << "\n-------Naive Bayes-------" << endl;

	//Calculate apriori probabilities
	float p_survived = 0.0;
	float p_died = 0.0;
	int countSurvived = 0;

	for(int i = 0; i < TRAINING_SIZE; i++){
		countSurvived += (int) x[i];
	}

	p_survived = (float)countSurvived / TRAINING_SIZE;
	p_died = 1 - p_survived;
	apriori_survived = p_survived;


	cout << "\n-------Apriori Probabilities-------" << endl;
	cout << "countSurvived: " << countSurvived << endl;
	cout << "p_survived: " << p_survived << endl;
	cout << "p_died: " << p_died << endl;


	cout << "\n-------Conditional Probabilities-------" << endl;

	//Calculate conditional probabilities
	//pclass
	for(int i = 0; i < TRAINING_SIZE; i++){
		if(x[i]){
			p_class_survived[0][w[i]-1]++;
		}
		else{
			p_class_survived[1][w[i]-1]++;
		}
	}

	for(int i = 0; i < 2; i++) 
		for(int j = 0; j < 3; j++) 
				p_class_survived[i][j] /= (i==0 ? countSurvived : TRAINING_SIZE - countSurvived);

	cout << "pclass: " << endl;
	cout << "\t\t\t1\t\t\t2\t\t\t3" << endl;
	cout << "Survived\t" << p_class_survived[0][0] << "\t" << p_class_survived[0][1] << "\t" << p_class_survived[0][2] << endl;
	cout << "Died\t\t" << p_class_survived[1][0] << "\t" << p_class_survived[1][1] << "\t\t" << p_class_survived[1][2] << endl;

	//sex
	for(int i = 0; i < TRAINING_SIZE; i++){
		if(x[i]){
			sex_survived[0][y[i]]++;
		}
		else{
			sex_survived[1][y[i]]++;
		}
	}

	for(int i = 0; i < 2; i++) 
		for(int j = 0; j < 2; j++) 
				sex_survived[i][j] /= (i==0 ? countSurvived : TRAINING_SIZE - countSurvived);

	cout << "sex: " << endl;
	cout << "\t\t\t1\t\t\t2" << endl;
	cout << "Survived\t" << sex_survived[0][0] << "\t" << sex_survived[0][1] << endl;
	cout << "Died\t\t" << sex_survived[1][0] << "\t" << sex_survived[1][1] << endl;

	//age
	vector<double> z_survived;
	vector<double> z_died;

	for(int i = 0; i < TRAINING_SIZE; i++){
		if(x[i]){
			z_survived.push_back(z[i]);
		}
		else{
			z_died.push_back(z[i]);
		}
	}

	age_survived[0][0] = mean(z_survived);
	age_survived[0][1] = var(z_survived);
	age_survived[1][0] = mean(z_died);
	age_survived[1][1] = var(z_died);

	cout << "age: " << endl;
	cout << "\t\t\t1\t\t\t2" << endl;
	cout << "Survived\t" << age_survived[0][0] << "\t\t" << age_survived[0][1] << endl;
	cout << "Died\t\t" << age_survived[1][0] << "\t\t" << age_survived[1][1] << endl;


}

void predict(vector<int> w, vector<bool> x, vector<bool> y, vector<double> z){
	int truePos = 0;
	int falsePos = 0;
	int trueNeg = 0;
	int falseNeg = 0;


	//Show the first 10 raw probabilities
	cout << "\n-------Raw Probabilities-------" << endl;
	cout << "\tSurvived\tDied" << endl;
	for(int i = TRAINING_SIZE; i < TRAINING_SIZE+10; i++){
		cout << "[" << i-TRAINING_SIZE << "]\t" << raw_prob(w[i], y[i], z[i]) << "\t" << 1-raw_prob(w[i], y[i], z[i]) << endl;
	}

	//Predict
	for(int i = TRAINING_SIZE; i < x.size(); i++){
		if(raw_prob(w[i], y[i], z[i]) > 0.5){
			if(x[i]){
				truePos++;
			}
			else{
				falsePos++;
			}
		}
		else{
			if(x[i]){
				falseNeg++;
			}
			else{
				trueNeg++;
			}
		}
	}

	int correct = truePos + trueNeg;
	int incorrect = falsePos + falseNeg;

	cout << "\nAccuracy: " << (float) (correct)/(correct+incorrect) << endl;
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

	high_resolution_clock::time_point start = high_resolution_clock::now();
	naive_bayes(pclass, survived, sex, age);
	high_resolution_clock::time_point stop = high_resolution_clock::now();
	predict(pclass, survived, sex, age);

	cout << "\n(Training took " << duration_cast<microseconds>(stop - start).count() << " microseconds)" << endl;
	cout << "Program terminated." << endl;

	return 0;
}