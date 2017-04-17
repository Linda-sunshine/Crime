package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import Analyzer.UserAnalyzer;

import opennlp.tools.util.InvalidFormatException;

public class MySplitMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 0; // Document length threshold

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String stopwords = "./data/Model/stopwords.dat";
			
//		String tweetTrain = "/if15/lg5bt/tweetData/tweetsTrain/";
//		String tweetTest = "/if15/lg5bt/tweetData/tweetsTest/";
		
		String data = "nongeo";
		String tweetDir = "./data/geo/tweets/";
		String tweetTrain = String.format("./data/%s/tweetSplit/tweetsTrain/", data);
		String tweetTest = String.format("./data/%s/tweetSplit/tweetsTest/", data);
			
		int k = 2000;
		String fv = "df";
		String type = "black";// "black" or "gay"
		String suffix = ".csv";
		String features = String.format("./data/%s/%s_%s_%d.txt", data, type, fv, k);
		
		String blackIAT = "./data/RaceIATCountyAgg.csv";
		String gayIAT = "./data/SexIATCountyAgg.csv";
		
		// trainSize: the size of counties used for testing.
		int trainSize = 50;
		System.out.println("Start splitting tweets into training and testing...");
		UserAnalyzer splitAnalyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
	
		// split the tweets data and iat data.
//		splitAnalyzer.loadIATCounties(blackIAT, trainSize);
//		splitAnalyzer.saveTrainTestTweets(tweetDir, suffix, tweetTrain, tweetTest);
		
		String blackTrainIAT = String.format("./data/%s/blackTrainIAT.csv",data);
		String blackTestIAT = String.format("./data/%s/blackTestIAT.csv", data);
		String gayTrainIAT = String.format("./data/%s/gayTrainIAT.csv", data);
		String gayTestIAT = String.format("./data/%s/gayTestIAT.csv", data);
//		splitAnalyzer.saveTrainTestIAT(blackIAT, blackTrainIAT, blackTestIAT);
//		splitAnalyzer.saveTrainTestIAT(gayIAT, gayTrainIAT, gayTestIAT);

	}
}
