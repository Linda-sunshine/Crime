package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Review;
import structures._Review.rType;
import structures._SparseFeature;
import structures._User;
import structures._stat;
import utils.Utils;

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

	public boolean loadCV(String filename){
		if (filename==null || filename.isEmpty())
			return false;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			Set<String> features = new HashSet<String>();
			m_Ngram = 2;//default value of Ngram
			String line;
			// collect all the features
			while ((line = reader.readLine()) != null) {
				features.add(SnowballStemming(Normalize(Tokenizer(line)[0])));
			}
			for(String fv: features)
				expandVocabulary(fv);
			reader.close();
			
			System.out.format("Load %d %d-gram seed words from %s...\n", m_featureNames.size(), m_Ngram, filename);
			m_isCVLoaded = true;
			return true;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
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
	public void loadUserDir(String folder){
		long t1 = System.currentTimeMillis();
		int count = 0;
		if(folder == null || folder.isEmpty())
			return;
		File dir = new File(folder);
		for(File f: dir.listFiles()){
			if(f.isFile()){
				loadUser(f.getAbsolutePath());
				count++;
				if(count%10 == 0)
					System.out.print(".");
			} else if (f.isDirectory())
				loadUserDir(f.getAbsolutePath());
		}
		
		long t2 = System.currentTimeMillis();
		t2 -= t1;
		t2 /= 1000;
		System.out.println(t2 + " secs are used to load the users.");
		System.out.format("%d users are loaded from %s...\n", count, folder);
	}
	
	String extractUserID(String text) {
		int index = text.indexOf('.');
		if (index==-1)
			return text;
		else
			return text.substring(0, index);
	}
	
	HashMap<String, _User> m_countyNameTweetsMap = new HashMap<String, _User>();
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
					
					if(AnalyzeDoc(tweet)) //Create the sparse vector for the review.
						tweets.add(tweet);	
				}
			}
			if(tweets.size() > 0)
				System.out.println(String.format("%d tweets are loaded from %s.\n", tweets.size(), filename));
			
			if(tweets.size() >= 1){//at least one for adaptation and one for testing
				_User cur = new _User(countyID, m_classNo, tweets);
				m_countyNameTweetsMap.put(countyID, cur);
				m_users.add(cur); //create new user from the file.
			} else if(tweets.size() == 1){// added by Lin, for those users with fewer than 2 reviews, ignore them.
				tweet = tweets.get(0);
				rollBack(Utils.revertSpVct(tweet.getSparse()), tweet.getYLabel());
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void loadIAT(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line, countyID;			
			String[] strs;

			// Skip the first line since it is user name.
			reader.readLine(); 

			while((line = reader.readLine()) != null){
				strs = line.replaceAll("\\s", "").split(",");
				countyID = findID(strs);
				if(m_countyNameTweetsMap.containsKey(countyID)){
					_User user = m_countyNameTweetsMap.get(countyID);
					user.setIATScore(Double.valueOf(strs[3]));
					user.setDemographics(strs);
				}
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
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
	
//	// save collected tweets.
//	public void saveTweets(String dir){
//		for(_User u: m_users){
//			
//		}
	}