package mains;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import Analyzer.UserAnalyzer;
import Classifier.supervised.IncidentPrediction;
import Classifier.supervised.LinearRegression;

import utils.Utils;
import weka.core.Instance;
import weka.core.Instances;

/*** The purpose of this function is to input data and the learned models, 
 * output the corresponding predicted values.
 * Input: features, learned models, test tweet data
 * Process: text processing of the tweets
 * Output: four predicted attitudes data.
 * @author lin
 */

public class IncidentMain {

	public static void main(String[] args) throws Exception{

		int k = 2000;
		boolean demo = false;
		String fv = "df";
		String blackImpModel = String.format("./data/models/black_model_imp_%s_%d_demo_%b.txt", fv, k, demo);
		String blackExpModel = String.format("./data/models/black_model_exp_%s_%d_demo_%b.txt", fv, k, demo);
		String gayImpModel = String.format("./data/models/gay_model_imp_%s_%d_demo_%b.txt", fv, k, demo);
		String gayExpModel = String.format("./data/models/gay_model_exp_%s_%d_demo_%b.txt", fv, k, demo);

		String prefix = "./data/";
		String data = "geo";
		
		/***Parameters related with analyzing the tweets.***/
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 0; // Document length threshold

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String stopwords = "./data/Model/stopwords.dat";
		
		String blackFeatures = String.format("%s/%s/black_%s_%d.txt", prefix, data, fv, k);
		String gayFeatures =  String.format("%s/%s/gay_%s_%d.txt", prefix, data, fv, k);
		
		String suffix = ".csv";
		String tweetTest = String.format("%s/%s/tweetSplit/tweetsTest/", prefix, data);
		String blackIAT = String.format("%s/%s/blackTestIAT.csv", prefix, data);
		String gayIAT = String.format("%s/%s/gayTestIAT.csv", prefix, data);
		
		String blackImpTestArff = String.format("%s/%s/ArffData/black_test_imp_%s_%d_demo_%b.arff", prefix, data, fv, k, demo);
		String blackExpTestArff = String.format("%s/%s/ArffData/black_test_exp_%s_%d_demo_%b.arff", prefix, data, fv, k, demo);
		String gayImpTestArff = String.format("%s/%s/ArffData/gay_test_imp_%s_%d_demo_%b.arff", prefix, data, fv, k, demo);
		String gayExpTestArff = String.format("%s/%s/ArffData/gay_test_exp_%s_%d_demo_%b.arff", prefix, data, fv, k, demo);

		/***Generate testing Arff files based on the selected features.***/
		System.out.println(String.format("Start generating %s testing tweets....", type));
		UserAnalyzer blackImp = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
		
		
		
		test_analyzer.loadUserDir(tweetTest, suffix);
		test_analyzer.loadIAT(testIAT);
		test_analyzer.setFeatureValues("TFIDF", 2);
		test_analyzer.generateArffData(testImpFile, "imp", demo);
		test_analyzer.generateArffData(testExpFile, "exp", demo);
		
		try{
			BufferedReader testImpReader = new BufferedReader(new FileReader(testFile));
			Instances impTest = new Instances(testImpReader);
			impTest.setClassIndex(impTest.numAttributes() - 1);
			IncidentPrediction inc = new IncidentPrediction();
			// load the weights.
			inc.loadWeights(modelFile);
			Instance ins;
			double trueY = 0, predY = 0; 
			for(int i=0; i<test.size(); i++){
				ins = test.get(i);
				trueY = ins.value(test.numAttributes()-1);
				predY = inc.classify(ins);
				System.out.print(String.format("TrueY:%.4f, predY by lr:%.4f", trueY, predY));
			}
		}catch(IOException e){
			e.printStackTrace();
		}
	}
}
