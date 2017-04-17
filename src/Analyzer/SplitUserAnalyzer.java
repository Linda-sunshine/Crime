package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._User;

public class SplitUserAnalyzer extends UserAnalyzer {
	HashMap<String, ArrayList<_User>> m_countyIDUsersMap = new HashMap<String, ArrayList<_User>>();

	public SplitUserAnalyzer(String tokenModel, int classNo, String providedCV,
			int Ngram, int threshold, boolean b) throws InvalidFormatException,
			FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, b);
	}
	
	// Extract split user id.
	String extractUserID(String text){
		int index = text.indexOf('.');
		if (index==-1){
			System.err.println("County name has wrong format!");
			return text;
		}
		String countyID = "";
		String[] strs = text.substring(0, index).split("_");
		countyID = strs[0];
		for(int i=1; i<strs.length-1; i++)
		 countyID += "_" + strs[i];
		return countyID;
	}
	
	// Load one file as a user here, multiple users may have the same countyID.
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
				if(!m_countyIDUsersMap.containsKey(countyID))
					m_countyIDUsersMap.put(countyID, new ArrayList<_User>());
					
				_User cur = new _User(countyID, m_classNo, tweets);
				m_countyIDUsersMap.get(countyID).add(cur); //create new user from the file.
				m_users.add(cur);
			} 
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	// load iat scores for the loaded counties/users, the splitted counties have the same iat score and demo info.
	public void loadIAT(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line, countyID;			
			String[] strs;

			// skip the first line
//			line = reader.readLine();
			int count = 0;
			while((line = reader.readLine()) != null){
				strs = line.replaceAll("\\s", "").split(",");
				countyID = findID(strs);
				if(m_countyIDUsersMap.containsKey(countyID)){
					for(_User user:m_countyIDUsersMap.get(countyID)){
						count++;
						user.setImpScore(Double.valueOf(strs[3]));
						user.setExpScore(Double.valueOf(strs[5]));
						user.setDemographics(strs);
					}
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
	
}
