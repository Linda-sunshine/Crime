package mains;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.Normalizer;
import java.util.ArrayList;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures._RankItem;
import utils.Utils;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import Analyzer.UserAnalyzer;

public class MSaveMain {

	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 0; // Document length threshold

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String stopwords = "./data/Model/stopwords.dat";
		
		int k = 2000;
		String prefix = "/if15/lg5bt/DSIData";//"./data"
//		String prefix = "./data";
		String data = "geo";
		String fv = "df";
		String type = "black";// "black" or "gay"
		String suffix = ".csv";
		boolean demo = false;// whether we include the demo in the training.
		
		String tweetTrain = String.format("%s/%s/tweetSplit/tweetsTrain/", prefix, data);
		String tweetTest = String.format("%s/%s/tweetSplit/tweetsTest/", prefix, data);
		
		String features = String.format("%s/%s/%s_%s_%d.txt", prefix, data, type, fv, k);
		
		String trainIAT = String.format("%s/%s/%sTrainIAT.csv", prefix, data, type);
		String testIAT = String.format("%s/%s/%sTestIAT.csv", prefix, data, type);
			
		String trainImpFile = String.format("%s/%s/ArffData/%s_train_imp_%s_%d.arff", prefix, data, type, fv, k);	
		String trainExpFile = String.format("%s/%s/ArffData/%s_train_exp_%s_%d.arff",prefix, data, type, fv, k);
		String testImpFile = String.format("%s/%s/ArffData/%s_test_imp_%s_%d.arff", prefix, data, type, fv, k);
		String testExpFile = String.format("%s/%s/ArffData/%s_test_exp_%s_%d.arff", prefix, data, type, fv, k);

		/***Feature selection based on DF.****/
//		int maxDF = -1, minDF = 0;
//		System.out.println(String.format("Start generating %s features based on DF....", type));
//		UserAnalyzer fs_analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
//		fs_analyzer.LoadStopwords(stopwords);
//		fs_analyzer.loadUserDir(tweetTrain, suffix);
//		fs_analyzer.featureSelection(features, "DF", maxDF, minDF, k);
//		
//		/***Generate training Arff files based on the selected features.***/
//		System.out.println(String.format("Start generating %s training tweets....", type));
//		UserAnalyzer train_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
//		train_analyzer.LoadStopwords(stopwords);
//		train_analyzer.loadUserDir(tweetTrain, suffix);
//		train_analyzer.loadIAT(trainIAT);
//		train_analyzer.setFeatureValues("TFIDF", 2);
//		train_analyzer.generateArffData(trainImpFile, "Imp", demo);
//		train_analyzer.generateArffData(trainExpFile, "Exp", demo);
//		
//		/***Generate testing Arff files based on the selected features.***/
//		System.out.println(String.format("Start generating %s testing tweets....", type));
//		UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
//		test_analyzer.loadUserDir(tweetTest, suffix);
//		test_analyzer.loadIAT(testIAT);
//		test_analyzer.setFeatureValues("TFIDF", 2);
//		test_analyzer.generateArffData(testImpFile, "Imp", demo);
//		test_analyzer.generateArffData(testExpFile, "Exp", demo);
		
		ArrayList<String> topFvs = new ArrayList<String>();
		double[] weights;
		MyPriorityQueue<_RankItem> rankq = new MyPriorityQueue<_RankItem>(k);
		
		/***Train linear regression models.***/
		try{
			/***Implicit attitudes.****/
			LinearRegression lr = new LinearRegression();
			System.out.println(String.format("Start loading %s training data from %s....", type, trainImpFile));
			BufferedReader trainReader = new BufferedReader(new FileReader(trainImpFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			System.out.print("Total number of attributes is "+train.numAttributes());
			lr.buildClassifier(train);

			System.out.println(String.format("Start loading %s testing data from %s....", type, testImpFile));
			BufferedReader testReader = new BufferedReader(new FileReader(testImpFile));
			Instances test = new Instances(testReader);
			test.setClassIndex(test.numAttributes() - 1);

			System.out.println("Start evaluation...");
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(lr, test);
			System.out.println(eval.toSummaryString("\nResults For Implicit Attitudes\n======\n", false));
		
			/***Write out the selected features.**/
			weights = lr.coefficients();
			for(int i=0; i<weights.length-1; i++){
				rankq.add(new _RankItem(i, Math.abs(weights[i])));
			}
			for(_RankItem it: rankq){
				if(it.m_value > 0){
					System.out.print(String.format("(%.3f,%d)\t", it.m_value, it.m_index));
					System.out.println( train.attribute(it.m_index).name());
					topFvs.add(train.attribute(it.m_index).name());
				}
			}
			System.out.println(topFvs.size() + " features are selected for imp attitudes.");
			try{
				PrintWriter writer = new PrintWriter(new File(String.format("%s/%s/%s_toplr_%d_imp.txt", prefix, data, type, k)));
				for(String f: topFvs)
					writer.write(f+"\n");
				writer.close();
			} catch(IOException e){
				e.printStackTrace();
			}
//			
//			/***Explicit attitudes.****/
//			System.out.println(String.format("Start loading %s training data from %s....", type, trainExpFile));
//			trainReader = new BufferedReader(new FileReader(trainExpFile));
//			train = new Instances(trainReader);
//			System.out.print("Total number of attributes is "+train.numAttributes());
//			train.setClassIndex(train.numAttributes() - 1);
//			lr.buildClassifier(train);
//		
//			System.out.println(String.format("Start loading %s testing data from %s....", type, testExpFile));
//			testReader = new BufferedReader(new FileReader(testExpFile));
//			test = new Instances(testReader);
//			test.setClassIndex(test.numAttributes() - 1);
//
//			System.out.println("Start evaluation...");
//			eval = new Evaluation(train);
//			eval.evaluateModel(lr, test);
//			System.out.println(eval.toSummaryString("\nResults for Explicit Results\n======\n", false));
//			
//			rankq.clear();
//			topFvs.clear();
//			/***Write out the selected features.**/
//			weights = lr.coefficients();
//			for(int i=0; i<weights.length-1; i++){
//				rankq.add(new _RankItem(i, Math.abs(weights[i])));
//			}
//			for(_RankItem it: rankq){
//				if(it.m_value > 0){
//					System.out.print(String.format("(%.3f,%d)\t", it.m_value, it.m_index));
//					System.out.println( train.attribute(it.m_index).name());
//					topFvs.add(train.attribute(it.m_index).name());
//				}
//			}
//			System.out.println(topFvs.size() + " features are selected for exp attitudes.");
//		
//			try{
//				PrintWriter writer = new PrintWriter(new File(String.format("%s/%s/%s_toplr_%d_exp.txt", prefix, data, type, k)));
//				for(String f: topFvs)
//					writer.write(f+"\n");
//				writer.close();
//			} catch(IOException e){
//				e.printStackTrace();
//			}
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}
}
