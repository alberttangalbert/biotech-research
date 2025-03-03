# Verdad Sage Research 
Factor generation and testing for systematic trading strategy.

## Python Environment Set-up 
0. Open Terminal (Mac/Linux) or Command Prompt (Windows)
    - Navigate to the desired parent directory where you want to clone the repository:
    ```
    $ cd /path/to/your/directory  # Replace with your actual path
    ```
    - Clone repository 
        - For Mac/Linux (using SSH):
            ```
            # git clone git@github.com:alberttangalbert/biotech-research.git
            ```
            Ensure you have SSH access set up with GitHub before using this method.
        - For Windows (using HTTPS):
            ```
            $ git clone https://github.com/alberttangalbert/biotech-research.git
            ```
            If using SSH on Windows, ensure you have an SSH key set up with GitHub and use the SSH version instead.
1. Make sure to have Python 3.13.1 installed 
    ```
    $ python3 --version
    Python 3.13.1
    ```
3. Create environment
   ```
   $ python3 -m venv venv
   ```
4. Activate the virtual environment
    > Linux/macOS
    ```
   $ source venv/bin/activate
   ```
   > Windows Git Bash 
   ```
   $ source venv/Scripts/activate
   ```
5. Install requirements
    ```
   $ pip install -r requirements.txt 
   ```
7. Add folders for storing data
    ```
    mkdir data/processed
    mkdir data/raw
    ```

## Retrive Data 
1. Configure Snowflake Environemnt
    - Log into the `PSA31288.us-east-1` Snowflake account
    - Navigate to `Projects` and create new SQL Worksheet
    - Select `Intern` for `Role` and `COMPUTE_WH` for `Run on Warehouse`
    - Select `BIOTECH_PROJECT` for `Databases` and `READ_ONLY` for `Schemas`
2. Closing prices 
    - Copy paste code from this [file](sql/fetch_price_close_usd.sql) and run
    - Verify there are 334.8K rows 
    - Download the results as a .csv file
    - Save the file in `data/raw` folder as `closing_prices.csv`
3. Sage factors 
    - Copy paste code from this [file](sql/fetch_sage_factors.sql) and run
    - Verify there are 124.1K rows 
    - Download the results as a .csv file
    - Save the file in `data/processed` folder as `sage_factors.csv`
3. Modalities and indications 
    - Copy paste code from this [file](sql/fetch_price_close_usd.sql) and run
    - Verify there are 1.7K rows 
    - Download the results as a .csv file
    - Save the file in `data/raw` folder as `mod_ind.csv`
4. Non-derivative insider transactions 
    - Copy paste code from this [file](sql/fetch_non_derivative_transactions.sql) and run
    - Verify there are 255.4K rows 
    - Download the results as a .csv file
    - Save the file in `data/raw` folder as `ndt.csv`
5. Market Cap
    - Copy paste code from this [file](sql/fetch_market_cap.sql) and run
    - Verify there are 143.3K rows 
    - Download the results as a .csv file
    - Save the file in `data/raw` folder as `market_cap.csv`

## Generate Factors 
0. Set-up Jupyter notebooks 
    - Select interpreter before running Jupyter notebooks 
        - Ctrl + Shift + P (Windows/Linux) or Cmd + Shift + P (Mac) to open the Command Palette.
        - Type "Python: Select Interpreter" and select the "venv" environment created in "Python Environment Set-up"
1. Uniqueness
    - Run all the code cells in this [notebook](notebooks/data_processing/uniqueness.ipynb)
    - It will output the file `data/processed/uniqueness.csv`
2. Modality + Indication Momentum 
    - Run all the code cells in this [notebook](notebooks/data_processing/mod_ind_momentum.ipynb)
    - It will output the file `data/processed/mod_ind_momentum.csv`
3. Residualized Short-Interest
    - Run all the code cells in this [notebook](notebooks/data_processing/short_interest_residualization.ipynb)
    - It will output the file `data/processed/residualized_short_interest.csv`

## Run Regression
Run the code cells in this [notebook](notebooks/data_analyses/regression.ipynb)

## Future Steps (Ranked by importance)
- Get 13F Filings 
    - WRDS has it, will pull 
- Create a sector wide momentum metric 
    - Returns or # of companies that exist over time
    - Will tell you about state of regulatory environemnt or biotech landscape
- Couple short interest data with other varaibles like institutional holdings, modalities, and indications
    - Put/Call Ratio Data See how much speculation there is 
    - Derivatives before FDA approval 
- Gather number of fda drugs approved in each modality/indication over time for each quarter 
- Generate a lot more granular data for indicaitons and modalities 
- Gather data on the number of deaths, injuries or sickness for each of these modalities or indications over time every month
- Gather data on the market cap / spending on each of these modalities or indications over time every month
- Number of other companies working on the same drug 
    - This will vastly improve results for Modality + Indication Momentum!
- Create scraping algorithm to eliminate look-ahead bias for modalities and indications
- Figure out how to integration insider transactions data 
- Look if a similar company has abnormal returns - pairs trading
- Find how Bitcoin prices correlate with biotech stocks
