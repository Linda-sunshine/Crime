package mains;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures._RankItem;
import utils.Utils;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import Analyzer.UserAnalyzer;
import Classifier.supervised.IncidentPrediction;

public class MPredictMain {

	public static String[] loadCountyNames(String filename){
		ArrayList<String> cs = new ArrayList<String>();
		if (filename==null || filename.isEmpty())
			return null;
			
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			String[] strs;
			while ((line = reader.readLine()) != null) {
				strs = line.split("\\.");
				cs.add(strs[0]);
			}
			reader.close();
			String[] counties = new String[cs.size()];
			counties = cs.toArray(counties);
			return counties;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return null;
		}
	}
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
	
		int k = 2000;
//		String prefix = "/if15/lg5bt/DSIData";//"./data"
		String prefix = "./data";
		String data = "geo";
		
		
		/***Parameters related with analyzing the tweets.***/
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 0; // Document length threshold
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		
//		for(String fv: new String[]{"toplr"}){
//			for(boolean demo: new boolean[]{false, true}){
//			for(String att: new String[]{"imp", "exp"}){
		String fv = "toplr";
		String type = "black";// "black" or "gay"
		String suffix = ".csv";
		String att = "imp";
		boolean demo = false;// whether we include the demo in the training.
		
//		String tweetTrain = String.format("%s/%s/tweetSplit/tweetsTrain/", prefix, data);
		String tweetTest = String.format("%s/%s/tweetSplit/tweetsTest/", prefix, data);
		String features = String.format("%s/%s/%s_%s_%d_%s.txt", prefix, data, type, fv, k, att);

//		String trainIAT = String.format("%s/%s/%sTrainIAT.csv", prefix, data, type);
		String testIAT = String.format("%s/%s/%sTestIAT.csv", prefix, data, type);
			
		String trainFile = String.format("%s/%s/ArffData/%s_train_%s_%s_%d_demo_%b.arff", prefix, data, type, att, fv, k, demo);	
		String testFile = String.format("%s/%s/ArffData/%s_test_%s_%s_%d_demo_%b.arff", prefix, data, type, att, fv, k, demo);
		String resFile = String.format("%s/%s/Results/%s_res_%s_%s_%d_demo_%b.txt", prefix, data, type, att, fv, k, demo);
		String modelFile = String.format("%s/models/%s_model_%s_%s_%d_demo_%b.txt", prefix, type, att, fv, k, demo);
		/***Train linear regression models.***/
		try{
			/***Implicit attitudes.****/
			LinearRegression lr = new LinearRegression();
			System.out.println(String.format("Start loading %s training data from %s....", type, trainFile));
			BufferedReader trainReader = new BufferedReader(new FileReader(trainFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			System.out.print("Total number of attributes is "+train.numAttributes());
			lr.buildClassifier(train);

//			UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
//			test_analyzer.loadUserDir(tweetTest, suffix);
//			test_analyzer.loadIAT(testIAT);
//			test_analyzer.setFeatureValues("TFIDF", 2);
//			test_analyzer.generateArffData(testFile, "imp", demo);
			
//			String blackImpFeatures = String.format("%s/%s/black_%s_%d_imp.txt", prefix, data, fv, k);
//			String blackImpTestArff = String.format("%s/%s/ArffData/black_test_imp_%s_%d_demo_%b.arff", prefix, data, fv, k, demo);

//			String tweetTest = String.format("%s/%s/tweetSplit/tweetsDebug/", prefix, data);
//			UserAnalyzer blackImpAnalyzer = new UserAnalyzer(tokenModel, classNumber, blackImpFeatures, Ngram, lengthThreshold, false);

			// generate black implicit test data
//			blackImpAnalyzer.loadUserDir(tweetTest, suffix);
//			blackImpAnalyzer.setFeatureValues("TFIDF", 2);
//			blackImpAnalyzer.generateArffData(blackImpTestArff, "imp", demo);
			
			System.out.println(String.format("Start loading %s testing data from %s....", type, testFile));
			BufferedReader testReader = new BufferedReader(new FileReader(testFile));
			Instances test = new Instances(testReader);
			test.setClassIndex(test.numAttributes() - 1);
			
			IncidentPrediction pred = new IncidentPrediction();
			pred.loadWeights(modelFile);
//			pred.setWeights(lr.coefficients());
			
			String[] counties = loadCountyNames("countyIndex.txt");
//			UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
//			analyzer.buildIATMap(testIAT, att);
//			HashMap<Double, String> iatMap = analyzer.getIATMap();
			Instance ins;
			double trueY = 0, predY_0 = 0, predY_1; 
			for(int i=0; i<test.size(); i++){
				ins = test.get(i);
				trueY = ins.value(test.numAttributes()-1);
				predY_0 = lr.classifyInstance(ins);
				predY_1 = pred.classify(ins);
				System.out.print(String.format("%s,true y: %.4f,pred by lr: %.4f, pred by self:%.4f\n",counties[i], trueY, predY_0, predY_1));
			}
		
//			System.out.println("Start evaluation...");
//			Evaluation eval = new Evaluation(train);
//			eval.evaluateModel(lr, test);
//			System.out.println(eval.toSummaryString("\nResults For Implicit Attitudes\n======\n", false));
		
			
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
//			System.out.println(topFvs.size() + " features are selected for imp attitudes.");
//			try{
//				PrintWriter writer = new PrintWriter(new File(String.format("%s/%s/%s_toplr_%d_imp.txt", prefix, data, type, k)));
//				for(String f: topFvs)
//					writer.write(f+"\n");
//				writer.close();
//			} catch(IOException e){
//				e.printStackTrace();
//			}
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
			e1.printStackTrace();
		}
//	}}}
	}
}
