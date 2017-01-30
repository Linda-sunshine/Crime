package mains;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import structures.LRParameter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import Analyzer.UserAnalyzer;

public class Execution {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		try{
			int classNumber = 2;
			int Ngram = 2; // The default value is unigram.
			int lengthThreshold = 0; // Document length threshold

			String tokenModel = "./data/Model/en-token.bin"; // Token model.
			String stopwords = "./data/Model/stopwords.dat";
			
			LRParameter param = new LRParameter(args);
			
			String tweetTrain = String.format("%s/%s/tweetSplit/tweetsTrain/", param.m_prefix, param.m_data);
			String tweetTest = String.format("%s/%s/tweetSplit/tweetsTest/", param.m_prefix, param.m_data);
						
			String trainIAT = String.format("%s/%s/%sTrainIAT.csv", param.m_prefix, param.m_data, param.m_type);
			String testIAT = String.format("%s/%s/%sTestIAT.csv", param.m_prefix, param.m_data, param.m_type);
			
			String features = String.format("%s/%s/%s_%s_%d_%s.txt", param.m_prefix, param.m_data, param.m_type, param.m_fv, param.m_k, param.m_att);
			String trainFile = String.format("%s/%s/ArffData/%s_train_%s_%s_%d_demo_%b.arff", param.m_prefix, param.m_data, param.m_type, param.m_att, param.m_fv, param.m_k, param.m_demo);	
			String testFile = String.format("%s/%s/ArffData/%s_test_%s_%s_%d_demo_%b.arff", param.m_prefix, param.m_data, param.m_type, param.m_att, param.m_fv, param.m_k, param.m_demo);
				
			System.out.print(String.format("[Info]k:%d,data:%s,fv:%s,type:%s,demo:%b,att:%s\n",param.m_k,param.m_data,param.m_fv,param.m_type,param.m_demo,param.m_att));
			/***Generate training Arff files based on the selected features.***/
			System.out.println(String.format("Start generating %s training tweets....", param.m_type));
			UserAnalyzer train_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
			train_analyzer.LoadStopwords(stopwords);
			train_analyzer.loadUserDir(tweetTrain, param.m_suffix);
			train_analyzer.loadIAT(trainIAT);
			train_analyzer.setFeatureValues("TFIDF", 2);
			train_analyzer.generateArffData(trainFile, param.m_att, param.m_demo);
			
			/***Generate testing Arff files based on the selected features.***/
			System.out.println(String.format("Start generating %s testing tweets....", param.m_type));
			UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
			test_analyzer.loadUserDir(tweetTest, param.m_suffix);
			test_analyzer.loadIAT(testIAT);
			test_analyzer.setFeatureValues("TFIDF", 2);
			test_analyzer.generateArffData(testFile, param.m_att, param.m_demo);
			
			LinearRegression lr = new LinearRegression();
			
			System.out.println(String.format("Start loading %s training data from %s....", param.m_type, trainFile));
			BufferedReader trainReader = new BufferedReader(new FileReader(trainFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			lr.buildClassifier(train);

			System.out.println(String.format("Start loading %s testing data from %s....", param.m_type, testFile));
			BufferedReader testReader = new BufferedReader(new FileReader(testFile));
			Instances test = new Instances(testReader);
			test.setClassIndex(test.numAttributes() - 1);

			System.out.println("Start evaluation...");
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(lr, test);
			System.out.println(eval.toSummaryString(String.format("\nResults For %s Attitudes\n======\n", param.m_att), false));

		} catch(Exception e1){
			e1.printStackTrace();
		}
	}
}
