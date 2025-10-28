"""
Automation Script for Diabetes Dataset Preprocessing
Author: Jessica
Description: Script untuk melakukan preprocessing otomatis pada diabetes dataset
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class DiabetesPreprocessor:
    """Class untuk melakukan preprocessing diabetes dataset secara otomatis"""
    
    def __init__(self, random_state=42):
        """
        Inisialisasi preprocessor
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state untuk reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        self.median_values = {}
        
    def load_data(self, filepath):
        """
        Load dataset dari file CSV
        
        Parameters:
        -----------
        filepath : str
            Path ke file CSV
            
        Returns:
        --------
        pd.DataFrame
            Dataset yang telah dimuat
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
            return df
        except FileNotFoundError:
            print(f"✗ Error: File {filepath} tidak ditemukan!")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error saat membaca file: {str(e)}")
            sys.exit(1)
    
    def drop_id_column(self, df):
        """
        Hapus kolom ID jika ada
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
            
        Returns:
        --------
        pd.DataFrame
            Dataset tanpa kolom ID
        """
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
            print("✓ Kolom 'Id' telah dihapus")
        return df
    
    def handle_zero_values(self, df):
        """
        Ganti nilai 0 dengan NaN untuk kolom tertentu
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
            
        Returns:
        --------
        pd.DataFrame
            Dataset dengan nilai 0 diganti NaN
        """
        zero_count = 0
        for col in self.cols_with_zero:
            if col in df.columns:
                count = (df[col] == 0).sum()
                zero_count += count
                df[col] = df[col].replace(0, np.nan)
        
        print(f"✓ {zero_count} nilai 0 telah diganti dengan NaN")
        return df
    
    def impute_missing_values(self, df):
        """
        Isi missing values dengan median
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
            
        Returns:
        --------
        pd.DataFrame
            Dataset dengan missing values ter-impute
        """
        missing_count = df.isnull().sum().sum()
        
        for col in self.cols_with_zero:
            if col in df.columns:
                median_value = df[col].median()
                self.median_values[col] = median_value
                df[col].fillna(median_value, inplace=True)
        
        print(f"✓ {missing_count} missing values telah di-impute dengan median")
        return df
    
    def handle_outliers(self, df):
        """
        Handle outliers menggunakan winsorizing (IQR method)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
            
        Returns:
        --------
        pd.DataFrame
            Dataset dengan outliers ter-handle
        """
        outliers_handled = 0
        
        for col in df.columns:
            if col != 'Outcome':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outliers_handled += outliers
                
                # Winsorizing
                df[col] = np.where(df[col] < lower_bound, lower_bound,
                          np.where(df[col] > upper_bound, upper_bound, df[col]))
        
        print(f"✓ {outliers_handled} outliers telah di-handle dengan winsorizing")
        return df
    
    def split_data(self, df, test_size=0.2):
        """
        Split data menjadi training dan testing set
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        test_size : float, default=0.2
            Proporsi data testing
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ Data split - Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features menggunakan StandardScaler
        
        Parameters:
        -----------
        X_train : pd.DataFrame atau np.array
            Data training
        X_test : pd.DataFrame atau np.array
            Data testing
            
        Returns:
        --------
        tuple
            (X_train_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✓ Features berhasil di-scale dengan StandardScaler")
        return X_train_scaled, X_test_scaled
    
    def apply_smote(self, X_train, y_train):
        """
        Apply SMOTE untuk handle class imbalance
        
        Parameters:
        -----------
        X_train : np.array
            Data training (sudah scaled)
        y_train : pd.Series atau np.array
            Target training
            
        Returns:
        --------
        tuple
            (X_train_resampled, y_train_resampled)
        """
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"✓ SMOTE applied - Samples: {len(y_train)} → {len(y_train_resampled)}")
        return X_train_resampled, y_train_resampled
    
    def save_preprocessed_data(self, df, output_path='diabetes_preprocessed.csv'):
        """
        Save preprocessed data ke CSV
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset yang sudah diproses
        output_path : str
            Path output file
        """
        df.to_csv(output_path, index=False)
        print(f"✓ Preprocessed data tersimpan di: {output_path}")
    
    def save_scaler(self, output_path='scaler.pkl'):
        """
        Save scaler object untuk production
        
        Parameters:
        -----------
        output_path : str
            Path output file
        """
        joblib.dump(self.scaler, output_path)
        print(f"✓ Scaler tersimpan di: {output_path}")
    
    def preprocess(self, input_filepath, output_filepath='diabetes_preprocessed.csv', 
                   save_scaler=True):
        """
        Main preprocessing pipeline
        
        Parameters:
        -----------
        input_filepath : str
            Path ke raw dataset
        output_filepath : str
            Path untuk menyimpan preprocessed dataset
        save_scaler : bool
            Apakah menyimpan scaler object
            
        Returns:
        --------
        dict
            Dictionary berisi hasil preprocessing
        """
        print("=" * 70)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 70)
        
        # 1. Load data
        df = self.load_data(input_filepath)
        
        # 2. Drop ID column
        df = self.drop_id_column(df)
        
        # 3. Handle zero values
        df = self.handle_zero_values(df)
        
        # 4. Impute missing values
        df = self.impute_missing_values(df)
        
        # 5. Handle outliers
        df = self.handle_outliers(df)
        
        # 6. Save preprocessed data
        self.save_preprocessed_data(df, output_filepath)
        
        # 7. Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # 8. Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # 9. Apply SMOTE
        X_train_resampled, y_train_resampled = self.apply_smote(X_train_scaled, y_train)
        
        # 10. Save scaler
        if save_scaler:
            self.save_scaler()
        
        print("=" * 70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_resampled': X_train_resampled,
            'y_train_resampled': y_train_resampled,
            'preprocessed_df': df
        }


def main():
    """
    Main function untuk menjalankan preprocessing
    """
    # Path ke raw dataset - file ada di root, script di preprocessing/
    input_path = '../diabetes_raw.csv'
    output_path = 'diabetes_preprocessing.csv'
    
    # Inisialisasi preprocessor
    preprocessor = DiabetesPreprocessor(random_state=42)
    
    # Jalankan preprocessing pipeline
    results = preprocessor.preprocess(input_path, output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Training samples (original): {len(results['y_train'])}")
    print(f"Training samples (after SMOTE): {len(results['y_train_resampled'])}")
    print(f"Testing samples: {len(results['y_test'])}")
    print(f"Number of features: {results['X_train'].shape[1]}")
    print(f"Preprocessed data saved: {output_path}")
    print(f"Scaler saved: scaler.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()
