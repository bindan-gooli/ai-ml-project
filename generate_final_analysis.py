import pandas as pd
import os

RESULTS_FILE = "updated_model_results.csv"

def generate_analysis():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Please wait for the pipeline to finish running.")
        return

    df = pd.read_csv(RESULTS_FILE)
    
    print("=========================================================")
    print("🚀 HYBRID MODEL FULL DATASET ANALYSIS REPORT 🚀")
    print("=========================================================\n")
    
    datasets = df['Dataset'].unique()
    
    for ds in sorted(datasets):
        ds_data = df[df['Dataset'] == ds]
        print(f"------------- Analysis for {ds} -------------")
        print(f"Total Combinations Evaluated: {len(ds_data)}")
        
        if len(ds_data) == 0:
            print("No data yet.\n")
            continue
            
        # Top model by F1 Score
        top_f1 = ds_data.loc[ds_data['F1_Score'].idxmax()]
        print(f"👑 Best Model by F1 Score: {top_f1['Hybrid_Combination']}")
        print(f"   -> F1 Score:  {top_f1['F1_Score']:.4f}")
        print(f"   -> AUC ROC:   {top_f1['AUC']:.4f}")
        print(f"   -> Precision: {top_f1['Precision']:.4f}")
        print(f"   -> Recall:    {top_f1['Recall']:.4f}")
        print(f"   -> Time:      {top_f1['Time_Seconds']/60:.2f} minutes\n")
        
        # Top model by AUC Score
        top_auc = ds_data.loc[ds_data['AUC'].idxmax()]
        print(f"🎯 Best Model by AUC Score: {top_auc['Hybrid_Combination']}")
        print(f"   -> AUC ROC:   {top_auc['AUC']:.4f}")
        print(f"   -> F1 Score:  {top_auc['F1_Score']:.4f}")
        print(f"   -> Time:      {top_auc['Time_Seconds']/60:.2f} minutes\n")
        
        # Fastest Model with acceptable performance (AUC > 0.90)
        acceptable = ds_data[ds_data['AUC'] > 0.90]
        if not acceptable.empty:
            fastest = acceptable.loc[acceptable['Time_Seconds'].idxmin()]
            print(f"⚡ Fastest Model (AUC > 0.90): {fastest['Hybrid_Combination']}")
            print(f"   -> Time:      {fastest['Time_Seconds']:.2f} seconds")
            print(f"   -> AUC ROC:   {fastest['AUC']:.4f}")
            print(f"   -> F1 Score:  {fastest['F1_Score']:.4f}\n")
            
        print("--------------------------------------------------\n")
        
    print("Analysis complete. You can view the full raw data in the generated Excel file.")

if __name__ == "__main__":
    generate_analysis()
