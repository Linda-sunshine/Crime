package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.UserAnalyzer;

public class MyPreprocessMain {

	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 0; // Document length threshold

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String stopwords = "./data/Model/stopwords.dat";
			
		String tweetTrain = "/if15/lg5bt/tweetData/tweetsTrain/";
		String tweetTest = "/if15/lg5bt/tweetData/tweetsTest/";
		
//		String tweetTrain = "./data/tweetsTrain_4/";
//		String tweetTest = "./data/tweetsTest_3/";
			
		int k = 2000;
		String fv = "df";
		String type = "black";// "black" or "gay"
		String suffix = ".csv";
		String features = String.format("./data/%s_%s_%d.txt", type, fv, k);
		
		String trainIAT = String.format("./data/%sTrain.csv", type);
		String testIAT = String.format("./data/%sTest.csv", type);
			
		String trainImpFile = String.format("/if15/lg5bt/ArffData/%s_train_imp_%s_%d_demo.arff", type, fv, k);	
		String trainExpFile = String.format("/if15/lg5bt/ArffData/%s_train_exp_%s_%d_demo.arff", type, fv, k);
		String testImpFile = String.format("/if15/lg5bt/ArffData/%s_test_imp_%s_%d_demo.arff", type, fv, k);
		String testExpFile = String.format("/if15/lg5bt/ArffData/%s_test_exp_%s_%d_demo.arff", type, fv, k);

//		int maxDF = -1, minDF = 0;
//		System.out.println(String.format("Start generating %s features based on DF....", type));
//		UserAnalyzer fs_analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
//		fs_analyzer.LoadStopwords(stopwords);
//		fs_analyzer.loadUserDir(tweetTrain, suffix);
//		fs_analyzer.featureSelection(features, "DF", maxDF, minDF, k);
		
//		System.out.println(String.format("Start generating %s training tweets....", type));
//		UserAnalyzer train_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
//		train_analyzer.LoadStopwords(stopwords);
//		train_analyzer.loadUserDir(tweetTrain, suffix);
//		train_analyzer.loadIAT(trainIAT);
//		train_analyzer.setFeatureValues("TFIDF", 2);
//		train_analyzer.generateArffData(trainImpFile, "Imp");
//		train_analyzer.generateArffData(trainExpFile, "Exp");
//		
//		System.out.println(String.format("Start generating %s testing tweets....", type));
//		UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
//		test_analyzer.loadUserDir(tweetTest, suffix);
//		test_analyzer.loadIAT(testIAT);
//		test_analyzer.setFeatureValues("TFIDF", 2);
//		test_analyzer.generateArffData(testImpFile, "Imp");
//		test_analyzer.generateArffData(testExpFile, "Exp");
		
		System.out.println(String.format("Start generating %s training tweets....", type));
		UserAnalyzer train_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
		train_analyzer.LoadStopwords(stopwords);
		train_analyzer.loadUserDir(tweetTrain, suffix);
		train_analyzer.loadIAT(trainIAT);
		train_analyzer.setFeatureValues("TFIDF", 2);
		train_analyzer.generateArffData(trainImpFile, "Imp");
		train_analyzer.generateArffData(trainExpFile, "Exp");
		
		System.out.println(String.format("Start generating %s testing tweets....", type));
		UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
		test_analyzer.loadUserDir(tweetTest, suffix);
		test_analyzer.loadIAT(testIAT);
		test_analyzer.setFeatureValues("TFIDF", 2);
		test_analyzer.generateArffData(testImpFile, "Imp");
		test_analyzer.generateArffData(testExpFile, "Exp");
	}
}
