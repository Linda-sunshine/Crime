package mains;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import Analyzer.UserAnalyzer;

public class MyPreprocessMain {

	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 0; // Document length threshold

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String stopwords = "./data/Model/stopwords.dat";
			
//		String tweetTrain = "/if15/lg5bt/tweetData/tweetsTrain/";
//		String tweetTest = "/if15/lg5bt/tweetData/tweetsTest/";
		
		String tweetTrain = "./data/tweetsTrain_4/";
		String tweetTest = "./data/tweetsTest_3/";
			
		int k = 1000;
		String fv = "seed"; //"fs", "demo"
		String type = "gay";// "black" or "gay"
		String suffix = ".csv";
		String features = String.format("./data/%s_%s.txt", type, fv);
		
		String trainIAT = String.format("./data/%sTrain.csv", type);
		String testIAT = String.format("./data/%sTest.csv", type);
			
		String trainFile = String.format("./data/%s_train_%s.arff", type, fv);		
		String testFile = String.format("./data/%s_test_%s.arff", type, fv);
		
//		System.out.println(String.format("Start generating %s features....", type));
//		UserAnalyzer fs_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
//		fs_analyzer.printFeatures();
//
//		fs_analyzer.LoadStopwords(stopwords);
//		fs_analyzer.loadUserDir(tweetTrain, suffix);
//		fs_analyzer.loadIAT(trainIAT);
//		fs_analyzer.setFeatureValues("TFIDF", 0);
//		fs_analyzer.generateArffData(trainFile);
//		
//		try{
//			// feature selection based on the feature coefficients.
//			LinearRegression lr = new LinearRegression();
//			BufferedReader trainReader = new BufferedReader(new FileReader(trainFile));
//			Instances train = new Instances(trainReader);
//			train.setClassIndex(train.numAttributes() - 1);
//			lr.buildClassifier(train);
//			fs_analyzer.selectFeatures(lr.coefficients(), k, features);
//		} catch(Exception e){
//			e.printStackTrace();
//		}
//		
		System.out.println(String.format("Start generating %s training tweets....", type));
		UserAnalyzer train_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
		train_analyzer.LoadStopwords(stopwords);
		train_analyzer.loadUserDir(tweetTrain, suffix);
		train_analyzer.loadIAT(trainIAT);
		train_analyzer.setFeatureValues("TFIDF", 0);
		train_analyzer.generateArffData(trainFile);
		
		System.out.println(String.format("Start generating %s testing tweets....", type));
		UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
		test_analyzer.loadUserDir(tweetTest, suffix);
		test_analyzer.loadIAT(testIAT);
		test_analyzer.setFeatureValues("TFIDF", 0);
		test_analyzer.generateArffData(testFile);
	}
}
