package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._User;

public class MacroUserAnalyzer extends UserAnalyzer{

	public MacroUserAnalyzer(String tokenModel, int classNo, String providedCV,
			int Ngram, int threshold, boolean b) throws InvalidFormatException,
			FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, b);
	}
	
	// Load each county's tweets and aggregate them as one macro tweet
	public void loadUser(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String countyID = extractUserID(file.getName()); //UserId is contained in the filename.
			if(m_countyIDUserMap.containsKey(countyID))
				System.err.println("The county exists in corpus!");
				
			// Skip the first line since it is user name.
			reader.readLine(); 
			String source = "";
			String[] strs;

			while((line = reader.readLine()) != null){
				strs = line.split(",");
				if(strs.length <6) 
					continue;
				source += ". " + strs[3];
			}
			_Review tweet = new _Review(m_corpus.getCollection().size(), source, 0, countyID);
			
			if(AnalyzeDoc(tweet)){
				_User cur = new _User(countyID, m_classNo, tweet);
				m_users.add(cur);
				m_countyIDUserMap.put(countyID, cur);
				m_corpus.addDoc(tweet);
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
