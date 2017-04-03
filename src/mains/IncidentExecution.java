package mains;

import java.io.BufferedReader;
import java.io.FileReader;

import structures.LRParameter;

import weka.core.Instance;
import weka.core.Instances;
import Analyzer.UserAnalyzer;
import Classifier.supervised.IncidentPrediction;

public class IncidentExecution {
	public static void main(String[] args) throws Exception{

		LRParameter param = new LRParameter(args);
		String blackImpModel = String.format("./data/models/black_model_imp_%s_%d_demo_%b.txt", param.m_fv, param.m_k, param.m_demo);
		String blackExpModel = String.format("./data/models/black_model_exp_%s_%d_demo_%b.txt", param.m_fv, param.m_k, param.m_demo);
		String gayImpModel = String.format("./data/models/gay_model_imp_%s_%d_demo_%b.txt", param.m_fv, param.m_k, param.m_demo);
		String gayExpModel = String.format("./data/models/gay_model_exp_%s_%d_demo_%b.txt", param.m_fv, param.m_k, param.m_demo);

		
		/***Parameters related with analyzing the tweets.***/
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 0; // Document length threshold

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String blackImpFeatures = String.format("%s/%s/black_%s_%d_imp.txt", param.m_prefix, param.m_data, param.m_fv, param.m_k);
		String blackExpFeatures = String.format("%s/%s/black_%s_%d_exp.txt", param.m_prefix, param.m_data, param.m_fv, param.m_k);
		String gayImpFeatures =  String.format("%s/%s/gay_%s_%d_imp.txt", param.m_prefix, param.m_data, param.m_fv, param.m_k);
		String gayExpFeatures =  String.format("%s/%s/gay_%s_%d_exp.txt", param.m_prefix, param.m_data, param.m_fv, param.m_k);

		String suffix = ".csv";
		String tweetTest = param.m_test;
//		String blackIAT = String.format("%s/%s/blackTestIAT.csv", prefix, data);
//		String gayIAT = String.format("%s/%s/gayTestIAT.csv", prefix, data);
		
		String blackImpTestArff = String.format("%s/%s/ArffData/black_test_imp_%s_%d_demo_%b.arff", param.m_prefix, param.m_data, param.m_fv, param.m_k, param.m_demo);
		String blackExpTestArff = String.format("%s/%s/ArffData/black_test_exp_%s_%d_demo_%b.arff", param.m_prefix, param.m_data, param.m_fv, param.m_k, param.m_demo);
		String gayImpTestArff = String.format("%s/%s/ArffData/gay_test_imp_%s_%d_demo_%b.arff", param.m_prefix, param.m_data, param.m_fv, param.m_k, param.m_demo);
		String gayExpTestArff = String.format("%s/%s/ArffData/gay_test_exp_%s_%d_demo_%b.arff", param.m_prefix, param.m_data, param.m_fv, param.m_k, param.m_demo);

		/***Generate testing Arff files based on the selected features.***/
		System.out.println("Start generating testing arffs...");
		UserAnalyzer blackImpAnalyzer = new UserAnalyzer(tokenModel, classNumber, blackImpFeatures, Ngram, lengthThreshold, false);
		UserAnalyzer blackExpAnalyzer = new UserAnalyzer(tokenModel, classNumber, blackExpFeatures, Ngram, lengthThreshold, false);
		UserAnalyzer gayImpAnalyzer = new UserAnalyzer(tokenModel, classNumber, gayImpFeatures, Ngram, lengthThreshold, false);
		UserAnalyzer gayExpAnalyzer = new UserAnalyzer(tokenModel, classNumber, gayExpFeatures, Ngram, lengthThreshold, false);

		// generate black implicit test data
		blackImpAnalyzer.loadUserDir(tweetTest, suffix);
		blackImpAnalyzer.setFeatureValues("TFIDF", 2);
		blackImpAnalyzer.generateArffData(blackImpTestArff, "imp", param.m_demo);
		
		// generate black explicit test data
		blackExpAnalyzer.loadUserDir(tweetTest, suffix);
		blackExpAnalyzer.setFeatureValues("TFIDF", 2);
		blackExpAnalyzer.generateArffData(blackExpTestArff, "exp", param.m_demo);
		
		// generate gay implicit test data
		gayImpAnalyzer.loadUserDir(tweetTest, suffix);
		gayImpAnalyzer.setFeatureValues("TFIDF", 2);
		gayImpAnalyzer.generateArffData(gayImpTestArff, "imp", param.m_demo);
		
		// generate gay explicit test data
		gayExpAnalyzer.loadUserDir(tweetTest, suffix);
		gayExpAnalyzer.setFeatureValues("TFIDF", 2);
		gayExpAnalyzer.generateArffData(gayExpTestArff, "exp", param.m_demo);
	
		/***Load learned model weights from files.***/
		// load the weights for black implicit attitude.
		BufferedReader blackImpReader = new BufferedReader(new FileReader(blackImpTestArff));
		Instances blackImpTest = new Instances(blackImpReader);
		blackImpTest.setClassIndex(blackImpTest.numAttributes() - 1);
		IncidentPrediction blackImpPred = new IncidentPrediction();
		blackImpPred.loadWeights(blackImpModel);
						
		// load the weights for black explicit attitude.
		BufferedReader blackExpReader = new BufferedReader(new FileReader(blackExpTestArff));
		Instances blackExpTest = new Instances(blackExpReader);
		blackExpTest.setClassIndex(blackExpTest.numAttributes() - 1);
		IncidentPrediction blackExpPred = new IncidentPrediction();
		blackExpPred.loadWeights(blackExpModel);

		// load the weights for gay implicit attitude.
		BufferedReader gayImpReader = new BufferedReader(new FileReader(gayImpTestArff));
		Instances gayImpTest = new Instances(gayImpReader);
		gayImpTest.setClassIndex(gayImpTest.numAttributes() - 1);
		IncidentPrediction gayImpPred = new IncidentPrediction();
		gayImpPred.loadWeights(gayImpModel);
			
		// load the weights for gay explicit attitude.
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
				
				System.out.print(String.format("BlackImp(%.4f,%.4f)\tBlackExp(%.4f,%.4f)\tGayImp(%.4f,%.4f)\tGayExp(%.4f,%.4f)\n", 
						blackImpIns.value(blackImpTest.numAttributes()-1), blackImpPy,
						blackExpIns.value(blackExpTest.numAttributes()-1), blackExpPy,
						gayImpIns.value(gayImpTest.numAttributes()-1), gayImpPy,
						gayExpIns.value(gayExpTest.numAttributes()-1), gayExpPy));
			}
		} else
			System.out.print(String.format("[Info] Different incident size! blackImp: %d, blackExp: %d, gayImp: %d, gayExp: %d\n", blackImpTest.size(), blackExpTest.size(), gayImpTest.size(), gayExpTest.size()));
	}
}

