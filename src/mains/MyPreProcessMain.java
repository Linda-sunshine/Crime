package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._User;
import Analyzer.Analyzer;
import Analyzer.CrossFeatureSelection;
import Analyzer.MultiThreadedUserAnalyzer;
import Analyzer.UserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.SVM;
import Classifier.supervised.modelAdaptation.HDP.MTCLRWithHDP;

public class MyPreProcessMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
	
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 2; // Document length threshold
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String providedCV = null;
		String tweetdir = "./data/tweets/";

		String stopwords = "./data/Model/stopwords.dat";
		String blackFv = "./data/black_features.txt";
		String gayFv = "./data/gay_features.txt";

//		/***Get statistics of the tweets of the seed words.*****/
//		UserAnalyzer black_analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
//		black_analyzer.LoadStopwords(stopwords);
//		black_analyzer.loadCV(blackFv);
//		black_analyzer.loadUserDir(tweetdir);
//		
//		UserAnalyzer gay_analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
//		gay_analyzer.LoadStopwords(stopwords);
//		gay_analyzer.loadCV(gayFv);
//		gay_analyzer.loadUserDir(tweetdir);
		
		/****split the tweets data into two folders: train and test.****/
		String traindir = "./data/train/tweetsTrain/";
		String testdir = "./data/test/tweetsTest/";
		
		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
		analyzer.splitData(tweetdir, traindir, testdir);
		
		/**Feature selection**/
		double startProb = 0.2; // Used in feature selection, the starting point of the features.
		double endProb = 1; // Used in feature selection, the ending point of the features.
		int maxDF = -1, minDF = 20; // Filter the features with DFs smaller than this threshold.
	}
}
