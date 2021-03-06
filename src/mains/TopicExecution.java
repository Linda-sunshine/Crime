package mains;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;

import structures.LRParameter;
import structures._Corpus;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA.pLSA_multithread;
import topicmodels.pLSA.pLSA;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import Analyzer.MacroUserAnalyzer;

public class TopicExecution {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		try{
			int classNumber = 2;
			int Ngram = 2; // The default value is unigram.
			int lengthThreshold = 0; // Document length threshold

			String tokenModel = "./data/Model/en-token.bin"; // Token model.
			String stopwords = "./data/Model/stopwords.dat";
				
			LRParameter param = new LRParameter(args);
			String suffix = ".csv";
				
			String testCounties = "./countyIndex.txt";
			String tweetDir = String.format("%s/%s/tweets/", param.m_prefix, param.m_data);
			String iat = String.format("%s/%s_IAT.csv", param.m_prefix, param.m_type);
			String features = String.format("%s/%s/%s_%s_%d_%s.txt", param.m_prefix, param.m_data, param.m_type, param.m_fv, param.m_k, param.m_att);
				
			// Topic modeling related parameters
			String topicmodel = "LDA_Gibbs"; // pLSA, LDA_Gibbs, LDA_Variational
			double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
			double converge = -1, lambda = 0.7; // negative converge means do need to check likelihood convergency
			int number_of_iteration = 100;
			boolean aspectSentiPrior = false;
				
			String trainFile = String.format("%s/%s/ArffData/%s_train_%s_%d_topic_%d_att_%s_demo_%b.arff", param.m_prefix, param.m_data, param.m_type, param.m_fv, param.m_k, param.m_number_of_topics, param.m_att, param.m_demo);	
			String testFile = String.format("%s/%s/ArffData/%s_test_%s_%d_topic_%d_att_%s_demo_%b.arff", param.m_prefix, param.m_data, param.m_type, param.m_fv, param.m_k, param.m_number_of_topics, param.m_att, param.m_demo);
					
			System.out.print(String.format("[Info]k:%d,data:%s,fv:%s,type:%s,demo:%b,att:%s\n",param.m_k,param.m_data,param.m_fv,param.m_type,param.m_demo,param.m_att));
				
			/***Generate training Arff files based on the selected features.***/
			System.out.println(String.format("Start generating %s topic vectors for tweets....", param.m_type));
			MacroUserAnalyzer analyzer = new MacroUserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
			analyzer.LoadStopwords(stopwords);
			analyzer.loadUserDir(tweetDir, suffix);
				
			analyzer.loadIAT(iat);
			analyzer.loadTestCounties(testCounties);			
			_Corpus c = analyzer.getCorpus(); // Get the collection of all the documents.

			pLSA tModel = null;
			if (topicmodel.equals("pLSA")) {			
				tModel = new pLSA_multithread(number_of_iteration, converge, beta, c, 
							lambda, param.m_number_of_topics, alpha);
			} else if (topicmodel.equals("LDA_Gibbs")) {		
				tModel = new LDA_Gibbs(number_of_iteration, converge, beta, c, 
					lambda, param.m_number_of_topics, alpha, 0.4, 50);
			}  else if (topicmodel.equals("LDA_Variational")) {		
				tModel = new LDA_Variational_multithread(number_of_iteration, converge, beta, c, 
						lambda, param.m_number_of_topics, alpha, 10, -1);
			} else {
				System.out.println("The selected topic model has not developed yet!");
				return;
			}
				
			tModel.setDisplayLap(0);
			tModel.setSentiAspectPrior(aspectSentiPrior);
			tModel.setInforWriter(String.format("../dsiData/%s_%s_%d_topic_%d_att_%s_demo_%b.txt", param.m_type, param.m_fv, param.m_k, param.m_number_of_topics, param.m_att, param.m_demo));
			tModel.EMonCorpus();	
			tModel.printTopWords(30);
				
			/***Generate train and test Arff files based on the selected features.***/
			System.out.println(String.format("Start generating %s training and testing data....", param.m_type));
			analyzer.setNumber4Topics(param.m_number_of_topics);
			analyzer.generateTopicArffData(trainFile, testFile, param.m_att, param.m_demo);

			System.out.println(String.format("Start loading %s training and testing data....", param.m_type));
			BufferedReader trainReader = new BufferedReader(new FileReader(trainFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
				
			LinearRegression lr = new LinearRegression();
			lr.buildClassifier(train);
				
			// Print out the trained weights
			System.out.println("[Info] Trained weights are as follows:");
			for(double v: lr.coefficients())
				System.out.print(v+"\t");
			System.out.println();
				
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
