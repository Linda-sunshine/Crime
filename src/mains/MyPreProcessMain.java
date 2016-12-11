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
			
		String tweetTrain = "./data/tweetsTrain/";
		String tweetTest = "./data/tweetsTest/";
		String suffix = ".csv";
			
		String blackSeed = "./data/black_seed.txt";
//		String blackFv = "./data/black_fetaures.txt";
			
		String blackTrainIAT = "./data/blackTrain.csv";
		String blackTestIAT = "./data/blackTest.csv";
			
//		/***Get statistics of the tweets of the seed words.*****/
//		System.out.println("Start analyzing black-related tweets....");
//	 	black_analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
//		black_analyzer.LoadStopwords(stopwords);
//		black_analyzer.loadCV(blackSeed);
//		black_analyzer.loadUserDir(tweetdir, suffix);
			
		String trainSeedFile = "./data/black_train_seed.arff";
//		String trainDemoFile = "./data/black_train_demo.arff";
//		String trainFsFile = "./data/black_train_fs.arff";
		
		String testSeedFile = "./data/black_test_seed.arff";
		
		System.out.println("Start generating black training tweets....");
		UserAnalyzer train_analyzer = new UserAnalyzer(tokenModel, classNumber, blackSeed, Ngram, lengthThreshold, false);
		train_analyzer.LoadStopwords(stopwords);
		train_analyzer.loadUserDir(tweetTrain, suffix);
		train_analyzer.loadIAT(blackTrainIAT);
		train_analyzer.setFeatureValues("TFIDF", 0);
		
		train_analyzer.generateArffData(trainSeedFile);
		
		System.out.println("Start generating black testing tweets....");
		UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, blackSeed, Ngram, lengthThreshold, false);
		test_analyzer.loadUserDir(tweetTest, suffix);
		test_analyzer.loadIAT(blackTestIAT);
		test_analyzer.setFeatureValues("TFIDF", 0);

		test_analyzer.generateArffData(testSeedFile);
		
	}

}
