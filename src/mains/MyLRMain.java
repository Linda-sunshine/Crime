package mains;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures._Doc;
import structures._RankItem;
import structures._User;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instances;
import Analyzer.Analyzer;
import Analyzer.CrossFeatureSelection;
import Analyzer.MultiThreadedUserAnalyzer;
import Analyzer.UserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.SVM;
import Classifier.supervised.modelAdaptation.HDP.MTCLRWithHDP;

public class MyLRMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		try{
			int classNumber = 2;
			int Ngram = 2; // The default value is unigram.
			int lengthThreshold = 0; // Document length threshold

			String tokenModel = "./data/Model/en-token.bin"; // Token model.
			String stopwords = "./data/Model/stopwords.dat";
			
			int k = 2000;
			String prefix = "/if15/lg5bt/DSIData";//"./data"
//			String prefix = "./data";
			
			String data = "geo";
			String fv = "df";
			String type = "black";// "black" or "gay"
			String suffix = ".csv";
			boolean demo = false;// whether we include the demo in the training.
			String att = "imp";// "exp"
			String tweetTrain = String.format("%s/%s/tweetSplit/tweetsTrain/", prefix, data);
			String tweetTest = String.format("%s/%s/tweetSplit/tweetsTest/", prefix, data);
						
			String trainIAT = String.format("%s/%s/%sTrainIAT.csv", prefix, data, type);
			String testIAT = String.format("%s/%s/%sTestIAT.csv", prefix, data, type);
			
			String features = String.format("%s/%s/%s_%s_%d_%s.txt", prefix, data, type, fv, k, att);
			String trainFile = String.format("%s/%s/ArffData/%s_train_%s_%s_%d_demo_%b.arff", prefix, data, type, att, fv, k, demo);	
			String testFile = String.format("%s/%s/ArffData/%s_test_%s_%s_%d_demo_%b.arff", prefix, data, type, att, fv, k, demo);
				
			/***Generate training Arff files based on the selected features.***/
			System.out.println(String.format("Start generating %s training tweets....", type));
			UserAnalyzer train_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
			train_analyzer.LoadStopwords(stopwords);
			train_analyzer.loadUserDir(tweetTrain, suffix);
			train_analyzer.loadIAT(trainIAT);
			train_analyzer.setFeatureValues("TFIDF", 2);
			train_analyzer.generateArffData(trainFile, att, demo);
			
			/***Generate testing Arff files based on the selected features.***/
			System.out.println(String.format("Start generating %s testing tweets....", type));
			UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
			test_analyzer.loadUserDir(tweetTest, suffix);
			test_analyzer.loadIAT(testIAT);
			test_analyzer.setFeatureValues("TFIDF", 2);
			test_analyzer.generateArffData(testFile, att, demo);
			
			LinearRegression lr = new LinearRegression();
			
			System.out.println(String.format("Start loading %s training data from %s....", type, trainFile));
			BufferedReader trainReader = new BufferedReader(new FileReader(trainFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			lr.buildClassifier(train);

			System.out.println(String.format("Start loading %s testing data from %s....", type, testFile));
			BufferedReader testReader = new BufferedReader(new FileReader(testFile));
			Instances test = new Instances(testReader);
			test.setClassIndex(test.numAttributes() - 1);

			System.out.println("Start evaluation...");
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(lr, test);
			System.out.println(eval.toSummaryString(String.format("\nResults For %s Attitudes\n======\n", att), false));
			
//			// Sort the weights of the learned features.
//			ArrayList<String> topFvs = new ArrayList<String>();
//			double[] weights = lr.coefficients();
//			MyPriorityQueue<_RankItem> rankq = new MyPriorityQueue<_RankItem>(k);
//			for(int i=0; i<weights.length; i++)
//				rankq.add(new _RankItem(i, Math.abs(weights[i])));
//			
//			for(_RankItem it: rankq){
//				if(it.m_value > 0){
//					System.out.print(String.format("(%.3f,%d)\t", it.m_value, it.m_index));
//					System.out.println( train.attribute(it.m_index).name());
//					topFvs.add(train.attribute(it.m_index).name());
//				}
//			}
//			System.out.print(String.format("%d features are selected for %s attitudes.\n", topFvs.size(), att));
//			try{
//				PrintWriter writer = new PrintWriter(new File(String.format("%s/%s/%s_toplr_%d_%s_demo_%b.txt", prefix, data, type, k, att, demo)));
//				for(String f: topFvs)
//					writer.write(f+"\n");
//				writer.close();
//			} catch(IOException e){
//				e.printStackTrace();
//			}
		} catch(Exception e1){
			e1.printStackTrace();
		}
	}
}

