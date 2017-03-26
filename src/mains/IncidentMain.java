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
		System.out.println("Start generating testing arffs...");
		UserAnalyzer blackAnalyzer = new UserAnalyzer(tokenModel, classNumber, blackFeatures, Ngram, lengthThreshold, false);
		UserAnalyzer gayAnalyzer = new UserAnalyzer(tokenModel, classNumber, gayFeatures, Ngram, lengthThreshold, false);

		blackAnalyzer.loadUserDir(tweetTest, suffix);
		blackAnalyzer.loadIAT(blackIAT);
		blackAnalyzer.setFeatureValues("TFIDF", 2);
		blackAnalyzer.generateArffData(blackImpTestArff, "imp", demo);
		blackAnalyzer.generateArffData(blackExpTestArff, "exp", demo);
		
		gayAnalyzer.loadUserDir(tweetTest, suffix);
		gayAnalyzer.loadIAT(gayIAT);
		gayAnalyzer.setFeatureValues("TFIDF", 2);
		gayAnalyzer.generateArffData(blackImpTestArff, "imp", demo);
		gayAnalyzer.generateArffData(blackExpTestArff, "exp", demo);
	
		// load the weights.
		BufferedReader blackImpReader = new BufferedReader(new FileReader(blackImpTestArff));
		Instances blackImpTest = new Instances(blackImpReader);
		blackImpTest.setClassIndex(blackImpTest.numAttributes() - 1);
		IncidentPrediction blackImpPred = new IncidentPrediction();
		blackImpPred.loadWeights(blackImpModel);
						
		BufferedReader blackExpReader = new BufferedReader(new FileReader(blackExpTestArff));
		Instances blackExpTest = new Instances(blackExpReader);
		blackExpTest.setClassIndex(blackExpTest.numAttributes() - 1);
		IncidentPrediction blackExpPred = new IncidentPrediction();
		blackExpPred.loadWeights(blackExpModel);

		BufferedReader gayImpReader = new BufferedReader(new FileReader(gayImpTestArff));
		Instances gayImpTest = new Instances(gayImpReader);
		gayImpTest.setClassIndex(gayImpTest.numAttributes() - 1);
		IncidentPrediction gayImpPred = new IncidentPrediction();
		gayImpPred.loadWeights(gayImpModel);
			
		BufferedReader gayExpReader = new BufferedReader(new FileReader(gayExpTestArff));
		Instances gayExpTest = new Instances(gayExpReader);
		gayExpTest.setClassIndex(gayExpTest.numAttributes() - 1);
		IncidentPrediction gayExpPred = new IncidentPrediction();
		gayExpPred.loadWeights(gayExpModel);
			
		int ttlSize = 0;
		double blackImpPy = 0, blackExpPy = 0, gayImpPy = 0, gayExpPy = 0; 
		Instance blackImpIns, blackExpIns, gayImpIns, gayExpIns;
		
		if(blackImpTest.size() == blackExpTest.size() && blackExpTest.size() == gayImpTest.size() && gayImpTest.size() == gayExpTest.size()){
			ttlSize = blackImpTest.size();
			for(int i=0; i<ttlSize; i++){
				blackImpIns = blackImpTest.get(i);
				blackExpIns = blackExpTest.get(i);
				gayImpIns = gayImpTest.get(i);
				gayExpIns = gayExpTest.get(i);
				
				blackImpPy = blackImpPred.classify(blackImpIns);
				blackExpPy = blackExpPred.classify(blackExpIns);
				gayImpPy = gayImpPred.classify(gayImpIns);
				gayExpPy = gayExpPred.classify(gayExpIns);
				
				System.out.print(String.format("BlackImp:(%.4f,%.4f)\tBlackExp(%.4f,%.4f)\tGayImp(%.4f,%.4f)\tGayExp(%.4f,%.4f)\n", 
						blackImpIns.value(blackImpIns.numAttributes()-1), blackImpPy,
						blackExpIns.value(blackExpIns.numAttributes()-1), blackExpPy,
						gayImpIns.value(gayImpIns.numAttributes()-1), gayImpPy,
						gayExpIns.value(gayExpIns.numAttributes()-1), gayExpPy));
			}
		}
	}
}
