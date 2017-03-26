package structures;

public class LRParameter {
	public String m_data = "geo";
	public String m_prefix = "./data";
	public String m_type = "black";
	public String m_fv = "toplr";//"df"
	public int m_k = 2000;
	public String m_att = "imp";//"exp"
	public boolean m_demo = false;
	public String m_suffix = ".csv";
//	public int m_userSet = 10; // The set of users we want to use.
//	public int m_ttlSizeSet = 24; // The total number of sizes.
//	public int m_ttlUserSetNo = 10; // The total number of user sets.
	
	public LRParameter(String argv[]){
		
		int i;
		
		//parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				exit_with_help();
			else if (argv[i-1].equals("-data"))
				m_data = argv[i];
			else if (argv[i-1].equals("-prefix"))
				m_prefix = argv[i];
			else if (argv[i-1].equals("-type"))
				m_type = argv[i];
			else if (argv[i-1].equals("-fv"))
				m_fv = argv[i];
			else if (argv[i-1].equals("-att"))
				m_att = argv[i];
			else if (argv[i-1].equals("-suffix"))
				m_suffix = argv[i];
			
			else if (argv[i-1].equals("-k"))
				m_k = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-demo"))
				m_demo = Boolean.valueOf(argv[i]);
			
			else
				exit_with_help();
		}
	}
	
	private void exit_with_help()
	{
		System.exit(1);
	}
}
