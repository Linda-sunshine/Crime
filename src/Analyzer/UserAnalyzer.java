package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._RankItem;
import structures._Review;
import structures._Review.rType;
import structures._SparseFeature;
import structures._User;
import structures._stat;
import utils.Utils;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 * 
 * @author Hongning Wang
 * Loading text file format for amazon and yelp reviews
 */
public class UserAnalyzer extends DocAnalyzer {
	
	ArrayList<_User> m_users; // Store all users with their reviews.
	double m_trainRatio = 0.25; // by default, the first 25% for training the global model 
	double m_adaptRatio = 0.5; // by default, the next 50% for adaptation, and rest 25% for testing
	int m_trainSize = 0, m_adaptSize = 0, m_testSize = 0;
	double m_pCount[] = new double[3]; // to count the positive ratio in train/adapt/test
	boolean m_enforceAdapt = false;
	// key: countyID, value: user
	HashMap<String, _User> m_countyIDUserMap = new HashMap<String, _User>();

	public UserAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, boolean b) 
			throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, Ngram, threshold, b);
		m_users = new ArrayList<_User>();
	}
	
	public void config(double train, double adapt, boolean enforceAdpt) {
		if (train<0 || train>1) {
			System.err.format("[Error]Incorrect setup of training ratio %.3f, which has to be in [0,1]\n", train);
			return;
		} else if (adapt<0 || adapt>1) {
			System.err.format("[Error]Incorrect setup of adaptation ratio %.3f, which has to be in [0,1]\n", adapt);
			return;
		} else if (train+adapt>1) {
			System.err.format("[Error]Incorrect setup of training and adaptation ratio (%.3f, %.3f), whose sum has to be in (0,1]\n", train, adapt);
			return;
		}
		
		m_trainRatio = train;
		m_adaptRatio = adapt;	
		m_enforceAdapt = enforceAdpt;
	}

	// Load the new cv.
	protected boolean loadNewCV(String filename){
		if (filename==null || filename.isEmpty())
			return false;
			
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			m_Ngram = 1;//default value of Ngram

			while ((line = reader.readLine()) != null) {
				if (line.startsWith("#")){//comments
					if (line.startsWith("#NGram")) {//has to be decoded
						int pos = line.indexOf(':');
						m_Ngram = Integer.valueOf(line.substring(pos+1));
					}						
				} else 
					expandVocabulary(line);
			}
			reader.close();
			System.out.format("Load %d %d-gram new features from %s...\n", m_featureNames.size(), m_Ngram, filename);
			m_isCVLoaded = true;
			return true;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
	}
	
	void setVocabStat(String term, int[] DFs) {
		_stat stat = m_featureStat.get(term);
		stat.setRawDF(DFs);
	}
	
	//Load all the users.
	public void loadUserDir(String folder, String suffix){
		long t1 = System.currentTimeMillis();
		int fCount = 0;
		if(folder == null || folder.isEmpty())
			return;
		File dir = new File(folder);
		for(File f: dir.listFiles()){
			if(f.isFile() && f.getAbsolutePath().endsWith(suffix)){
				loadUser(f.getAbsolutePath());
				fCount++;
				if(fCount%10 == 0)
					System.out.print(".");
			} else if (f.isDirectory())
				loadUserDir(f.getAbsolutePath(), suffix);
		}
		
		long t2 = System.currentTimeMillis();
		t2 -= t1;
		t2 /= 1000;
		System.out.println("---------------------------------------------");
		System.out.println(t2 + " secs are used to load the users.");
		System.out.format("%d/%d users are loaded from %s...\n", m_users.size(), fCount, folder);
	}
	
	String extractUserID(String text) {
		int index = text.indexOf('.');
		if (index==-1)
			return text;
		else
			return text.substring(0, index);
	}
	
	// Load one file as a user here. 
	public void loadUser(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String countyID = extractUserID(file.getName()); //UserId is contained in the filename.
			
			// Skip the first line since it is user name.
			reader.readLine(); 
			_Review tweet;
			String source;
			String[] strs;
			ArrayList<_Review> tweets = new ArrayList<_Review>();
			while((line = reader.readLine()) != null){
				strs = line.split(",");
				if(strs.length <6) 
					continue;
				else{
					source = strs[3];
					tweet = new _Review(m_corpus.getCollection().size(), source, 0, countyID);
					
					if(AnalyzeDoc(tweet))//Create the sparse vector for the review.
						tweets.add(tweet);
					
				}
			}
			if(tweets.size() >= 1){
				if(!m_countyIDUserMap.containsKey(countyID)){
					_User cur = new _User(countyID, m_classNo, tweets);
					m_countyIDUserMap.put(countyID, cur);
					m_users.add(cur); //create new user from the file.
				} else{
					System.err.println("The county exists in the map!");
				}
			} 
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	// load iat scores for the loaded counties/users.
	public void loadIAT(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line, countyID;			
			String[] strs;

			// skip the first line
			line = reader.readLine();
			int count = 0;
			while((line = reader.readLine()) != null){
				strs = line.replaceAll("\\s", "").split(",");
				countyID = findID(strs);
				if(m_countyIDUserMap.containsKey(countyID)){
					count++;
					_User user = m_countyIDUserMap.get(countyID);
					user.setImpScore(Double.valueOf(strs[3]));
					user.setExpScore(Double.valueOf(strs[5]));
					user.setDemographics(strs);
				}
				else
					System.out.println(countyID);
			}
			System.out.println("the number of counties of iat score "+count);
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	// Test counties' names
	ArrayList<String> m_trainCounties = new ArrayList<String>();
	ArrayList<String> m_testCounties = new ArrayList<String>();
	// Load the test counties' names
	public void loadTestCounties(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line, countyID;			
			while((line = reader.readLine()) != null){
				countyID = extractUserID(line);
				m_testCounties.add(countyID);
			}
			reader.close();
			for(String cname: m_countyIDUserMap.keySet()){
				if(!m_testCounties.contains(cname))
					m_trainCounties.add(cname);
			}
			System.out.println(m_trainCounties.size() + " train county names are loaded!");
			System.out.println(m_testCounties.size() + " test county names are loaded!");
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
//	HashMap<Double, String> m_iatCountyMap = new HashMap<Double, String>();
//	DecimalFormat m_df = new DecimalFormat("#.0000");
//	public void buildIATMap(String filename, String att){
//		try {
//			File file = new File(filename);
//			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
//			String line, countyID;			
//			String[] strs;
//
//			// Skip the first line since it is user name.
////			reader.readLine(); 
//			int count = 0;
//			double iat = 0;
//			
//			while((line = reader.readLine()) != null){
//				strs = line.replaceAll("\\s", "").split(",");
//				
//				if(att.equals("imp"))
//					iat = Double.valueOf(m_df.format(Double.valueOf(strs[3])));
//				else if(att.equals("exp"))
//					iat = Double.valueOf(m_df.format(Double.valueOf(strs[5])));
//				else
//					System.out.println("Attitude not supported!");
//				
//				countyID = findID(strs);
//				if(!m_iatCountyMap.containsKey(iat)){
//					count++;
//					m_iatCountyMap.put(iat, countyID);
//				} else
//					System.out.println(countyID + " The same attitude value!");
//			}
//			System.out.println("the number of counties of iat score "+count);
//			reader.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}
//	public HashMap<Double, String> getIATMap(){
//		return m_iatCountyMap;
//	}
//	ArrayList<String> m_counties = new ArrayList<String>();
//	HashSet<String> m_sltCounties = new HashSet<String>();
//	public void loadIATCounties(String filename, int k){
//		try {
//			File file = new File(filename);
//			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
//			String line, countyID;			
//			String[] strs;
//
//			// Skip the first line since it is user name.
//			reader.readLine(); 
//
//			while((line = reader.readLine()) != null){
//				strs = line.replaceAll("\\s", "").split(",");
//				countyID = findID(strs);
//				m_counties.add(countyID);
//			}
//			reader.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//		int idx = 0;
//		while(m_sltCounties.size() < k){
//			idx = (int)(Math.random()*m_counties.size());
//			m_sltCounties.add(m_counties.get(idx));
//		}
//	}
//	HashSet<String> tweetCounties = new HashSet<String>();
//	HashSet<String> iatCounties = new HashSet<String>();
//	public void saveTrainTestTweets(String folder, String suffix, String traindir, String testdir){
//		if(folder == null || folder.isEmpty())
//			return;
//		File dir = new File(folder);
//		String userID = "";
//		int trainCount = 0, testCount = 0;
//		try{			
//			for(File f: dir.listFiles()){
//				if(f.isFile() && f.getAbsolutePath().endsWith(suffix)){
//					userID  = extractUserID(f.getName());
//					tweetCounties.add(userID);
//					if(m_sltCounties.contains(userID)){
//						Files.copy(f.toPath(), new File(testdir+f.getName()).toPath());
//						testCount++;
//						if(testCount%10 == 0)
//							System.out.print("o");
//					} else{
//						Files.copy(f.toPath(), new File(traindir+f.getName()).toPath());
//						trainCount++;
//						if(trainCount%10 == 0)
//							System.out.print("x");
//					}	
//				} else{
//					System.out.println(f.getName());
//					System.out.println("Wrong format in saving training and testing tweets.");
//				}
//			}
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}
	public String findID(String[] strs){
		// remove the left "
		strs[0] = strs[0].substring(1, strs[0].length());
		// remove the right "
		strs[1] = strs[1].substring(0, strs[1].length()-1);
		String res = strs[0].toLowerCase()+"_"+strs[1].trim().toLowerCase();
		return res;
	}
	//[0, train) for training purpose
	//[train, adapt) for adaptation purpose
	//[adapt, 1] for testing purpose
	void allocateReviews(ArrayList<_Review> reviews) {
		Collections.sort(reviews);// sort the reviews by timestamp
		int train = (int)(reviews.size() * m_trainRatio), adapt;
		if (m_enforceAdapt)
			adapt = Math.max(1, (int)(reviews.size() * (m_trainRatio + m_adaptRatio)));
		else
			adapt = (int)(reviews.size() * (m_trainRatio + m_adaptRatio));
		
		_Review r;
		for(int i=0; i<reviews.size(); i++) {
			r = reviews.get(i);
			if (i<train) {
				r.setType(rType.TRAIN);
				if (r.getYLabel()==1)
					m_pCount[0] ++;
				
				m_trainSize ++;
			} else if (i<adapt) {
				r.setType(rType.ADAPTATION);
				if (r.getYLabel()==1)
					m_pCount[1] ++;
				
				m_adaptSize ++;
			} else {
				r.setType(rType.TEST);
				if (r.getYLabel()==1)
					m_pCount[2] ++;
				
				m_testSize ++;
			}
		}
	}

	//Return all the users.
	public ArrayList<_User> getUsers(){
		System.out.format("[Info]Training size: %d(%.2f), adaptation size: %d(%.2f), and testing size: %d(%.2f)\n", 
				m_trainSize, m_trainSize>0?m_pCount[0]/m_trainSize:0.0,
				m_adaptSize, m_adaptSize>0?m_pCount[1]/m_adaptSize:0.0,
				m_testSize, m_testSize>0?m_pCount[2]/m_testSize:0.0);
		return m_users;
	}
	
	public Collection<_Doc> mergeReviews(){
		Collection<_Doc> rvws = new ArrayList<_Doc>();
		for(_User u: m_users){
			rvws.addAll(u.getReviews());
		}
		return rvws;
	}

//	
//	// Split the twitter data into half as training and half as testing.
//	public void splitData(String folder, String suffix, String traindir, String testdir){
//		long t1 = System.currentTimeMillis();
//		int count = 0;
//		if(folder == null || folder.isEmpty())
//				return;
//		File dir = new File(folder);
//		for(File f: dir.listFiles()){
//			if(f.isFile() && f.getAbsolutePath().endsWith(".csv")){
//				splitUser(f.getAbsolutePath(), traindir, testdir);
//				count++;
//				if(count%10 == 0)
//					System.out.print(".");
//				} else if (f.isDirectory())
//					loadUserDir(f.getAbsolutePath(), suffix);
//		}
//			
//		long t2 = System.currentTimeMillis();
//		t2 -= t1;
//		t2 /= 1000;
//		System.out.println(t2 + " secs are used to load the tweets.");
//		System.out.format("%d counties of tweets are splitted from %s...\n", count, folder);
//	}
	
	// split the tweets of one county into two parts.
	public void splitUser(String filename, String traindir, String testdir){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			
			String countyID = extractUserID(file.getName()); //UserId is contained in the filename.
			String trainfile = traindir + countyID + ".csv";
			String testfile = testdir + countyID + ".csv";
			PrintWriter trainWriter = new PrintWriter(new File(trainfile));
			PrintWriter testWriter = new PrintWriter(new File(testfile));
			int trainCount = 0, testCount = 0;
			// Skip the first line since it is user name.
			reader.readLine(); 
	
			// Read Raw tweets.
			ArrayList<String> rawTweets = new ArrayList<String>();
			HashSet<Integer> trainIndexes = new HashSet<Integer>();
			while((line = reader.readLine()) != null){
				rawTweets.add(line);
			}
			
			// If the tweets size is too small, it is hard to get half of them randomly.
			if(rawTweets.size() < 100){
				for(int i=0; i<rawTweets.size(); i+=2){
					trainWriter.write(rawTweets.get(i)+"\n");
					if(i+1 < rawTweets.size())
						testWriter.write(rawTweets.get(i+1)+"\n");
				}
				trainCount = rawTweets.size()/2;
				testCount = rawTweets.size() - trainCount;
			} else{
				boolean[] trainFlags = new boolean[rawTweets.size()];
				// select random tweets as the training data
				while(trainIndexes.size() < rawTweets.size()/2){
					int index = (int)(Math.random()*rawTweets.size());
					trainIndexes.add(index);
					trainFlags[index] = true;
				}
				for(int i=0; i<trainFlags.length; i++){
					if(trainFlags[i])
						trainWriter.write(rawTweets.get(i)+"\n");
					else
						testWriter.write(rawTweets.get(i)+"\n");
				}
				trainCount = trainIndexes.size();
				testCount = rawTweets.size() - trainCount;
			}
			System.out.println(String.format("%s\ttrain\t%d\ttest\t%d\n", countyID, trainCount, testCount));
			reader.close();
			trainWriter.close();
			testWriter.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	public String extractRelation(String filename){
		String[] strs = filename.split("/");
		String name = strs[strs.length-1];
		return name.substring(0, name.indexOf("."));
	}
	String[] m_demos = new String[]{"avgAge", "pctHisp", "pctWht", "pctBlck", "pctFml", "pctCllg", "pctCnsv", "pctUnemp"};
	
	// Generate the data in arff format for linear regression.
	public void generateArffData(String filename, String att, boolean demo){
		PrintWriter writer;
		try{
			writer = new PrintWriter(new File(filename));
			writer.write(String.format("@RELATION %s\n", extractRelation(filename)));
			/**Write attributes:
			 * the first one is bias, the following ones are features, the final one is y.**/
			writer.write("@ATTRIBUTE bias\tNUMERIC\n");
			for(String s: m_featureNames){
				writer.write(String.format("@ATTRIBUTE %s\tNUMERIC\n", s));
			}
			if(demo){
				for(String s: m_demos)
					writer.write(String.format("@ATTRIBUTE %s\tNUMERIC\n", s));
			}
			writer.write("@ATTRIBUTE ylabel\tNUMERIC\n\n@Data\n");
			for(_User u: m_users){
				writer.write("{");
				double[] vct = demo ? formatEachUserDemo(u, att) : formatEachUser(u, att);
				for(int i=0; i<vct.length-1; i++){
					if(vct[i] != 0)
						writer.write(String.format("%d %.4f,", i, vct[i]));
				}
				writer.write(String.format("%d %.4f", vct.length-1, vct[vct.length-1]));
				writer.write("}\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	// the number of topics in our problem
	int m_number_of_topics = 0;
	public void setNumber4Topics(int nu){
		m_number_of_topics = nu;
	}
	// Generate train and test files with the topic models.
	public void generateTopicArffData(String train, String test, String att, boolean demo){
		ArrayList<_User> trainUsers = new ArrayList<_User>();
		ArrayList<_User> testUsers = new ArrayList<_User>();
		for(_User u: m_users){
			if(m_testCounties.contains(u.getUserID()))
				testUsers.add(u);
			else
				trainUsers.add(u);
		}
		generateOneTopicArffData(trainUsers, train, att, demo);
		generateOneTopicArffData(testUsers, test, att, demo);
	}

// Generate the topic vectors in arff format for linear regression.
	public void generateOneTopicArffData(ArrayList<_User> users, String filename, String att, boolean demo){
		PrintWriter writer;
		try{
			writer = new PrintWriter(new File(filename));
			writer.write(String.format("@RELATION %s\n", extractRelation(filename)));
			for(int i=0; i<m_number_of_topics; i++){
				writer.write(String.format("@ATTRIBUTE topic_%d\tNUMERIC\n", i));
			}
			if(demo){
				for(String s: m_demos)
					writer.write(String.format("@ATTRIBUTE %s\tNUMERIC\n", s));
			}
			writer.write("@ATTRIBUTE ylabel\tNUMERIC\n\n@Data\n");
			for(_User u: users){
				writer.write("{");
				double[] vct = demo ? formatEachUserTopicDemo(u, att) : formatEachUserTopic(u, att);
				for(int i=0; i<vct.length; i++){
					if(vct[i] != 0){
						writer.write(String.format("%d %.4f", i, vct[i]));
						if(i != vct.length-1)
							writer.write(",");
					}
				}
				writer.write("}\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void selectFeatures(double[] coef, int k, String filename){
		if(k > coef.length)
			System.out.println("K out of feature range!");
		_RankItem[] items = new _RankItem[coef.length];
		for(int i=0; i<coef.length; i++){
			items[i] = new _RankItem(i, Math.abs(coef[i]));
		}
		System.out.print(String.format("[Info] %sd features in total!\n", coef.length));
		Arrays.sort(items, new Comparator<_RankItem>(){
			@Override
			public int compare(_RankItem i1, _RankItem i2){
				return (int) (i2.m_value - i1.m_value);
			}
		});
		String[] features = new String[k];
		for(int i=0; i<k; i++){
			_RankItem ri = items[i];
			if(ri.m_index == 0)
				features[i] = "BIAS";
			else
				features[i] = m_featureNames.get(ri.m_index-1);
		}
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			for(String s: features)
				writer.write(s+"\n");
			System.out.print(String.format("%d features are written to %s", k, filename));
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void printFeatures(){
		for(String s: m_featureNames)
			System.out.print(s+",");
	}
	// Format the user tweets with/without demo
	public double[] formatEachUser(_User u, String att){
		double[] vct = new double[getFeatureSize()+2];
		for(_Review r: u.getReviews()){
			for(_SparseFeature sf: r.getSparse()){
				vct[sf.getIndex()+1] += sf.getValue();
			}
		}
		Utils.L2Normalization(vct);
		if(att.equals("imp"))
			vct[vct.length-1] = u.getImpScore();
		else if(att.equals("exp"))
			vct[vct.length-1] = u.getExpScore();
		return vct;
	}
	public double[] formatEachUserDemo(_User u, String att){
		double[] vct = new double[getFeatureSize()+10];// 8 features for demo
		for(_Review r: u.getReviews()){
			for(_SparseFeature sf: r.getSparse()){
				vct[sf.getIndex()+1] += sf.getValue();
			}
		}
		double[] demo = u.getDemographics();
		for(int i=0; i<demo.length; i++){
			vct[getFeatureSize() + 1 + i] = demo[i];
		}
		Utils.L2Normalization(vct);
		if(att.equals("imp"))
			vct[vct.length-1] = u.getImpScore();
		else if(att.equals("exp"))
			vct[vct.length-1] = u.getExpScore();
		return vct;
	}
	// Format the user topic vector with/without demo
	public double[] formatEachUserTopic(_User u, String att){
		double[] vct = new double[m_number_of_topics+1];
		if(u.getReviews().size() == 1){
			_Review r = u.getReviews().get(0);
			double[] topics = r.getTopics();
			for(int i=0; i< topics.length; i++){
				vct[i] = topics[i];
			}
		} else{
			System.err.println("Wrong review size for each county!");
		}
		if(att.equals("imp"))
			vct[vct.length-1] = u.getImpScore();
		else if(att.equals("exp"))
			vct[vct.length-1] = u.getExpScore();
		return vct;
	}
	// With demographics
	public double[] formatEachUserTopicDemo(_User u, String att){
		double[] vct = new double[getFeatureSize()+9];// 8 features for demo
		if(u.getReviews().size() == 1){
			_Review r = u.getReviews().get(0);
			double[] topics = r.getTopics();
			for(int i=0; i< topics.length; i++){
				vct[i] = topics[i];
			}
		} else{
			System.err.println("Wrong review size for each county!");
		}
		double[] demo = u.getDemographics();
		for(int i=0; i<demo.length; i++){
			vct[m_number_of_topics + i] = demo[i];
		}
		Utils.L2Normalization(vct);
		if(att.equals("imp"))
			vct[vct.length-1] = u.getImpScore();
		else if(att.equals("exp"))
			vct[vct.length-1] = u.getExpScore();
		return vct;
	}
	
//	// Save the IAT files into train/test files.
//	public void saveTrainTestIAT(String iat, String trainIAT, String testIAT){
//		try {
//			File file = new File(iat);
//			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
//			String line, countyID;			
//			String[] strs;
//			ArrayList<String> train = new ArrayList<String>();
//			ArrayList<String> test = new ArrayList<String>();
//			// Skip the first line since it is user name.
//			reader.readLine(); 
//
//			while((line = reader.readLine()) != null){
//				strs = line.replaceAll("\\s", "").split(",");
//				countyID = findID(strs);
//				iatCounties.add(countyID);
//				if(m_sltCounties.contains(countyID))
//					test.add(line);
//				else
//					train.add(line);
//			}
//			reader.close();
//			int count = 0;
//			for(String ic: iatCounties){
//				if(tweetCounties.contains(ic))
//					count++;
//				else
//					System.out.println(ic + " is missing.");
//			}
//			System.out.println(count);
//			// write out the train iats.
//			PrintWriter writer = new PrintWriter(new File(trainIAT));
//			for(String s: train)
//				writer.write(s+"\n");
//			writer.close();
//			// writer out the test iats.
//			writer = new PrintWriter(new File(testIAT));
//			for(String s: test)
//				writer.write(s+"\n");
//			writer.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}
}
