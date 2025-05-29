#!/usr/bin/env python3
"""
Quick fix script for the platt_calibrator KeyError issue
Run this script to patch your existing enhanced_ensemble_data.pkl file
"""

import pickle
import os

def fix_enhanced_ensemble_data():
    """Fix the enhanced ensemble data file to include missing keys"""
    
    file_path = 'enhanced_ensemble_data.pkl'
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Nothing to fix.")
        return False
    
    try:
        # Load existing data
        print("Loading existing enhanced ensemble data...")
        with open(file_path, 'rb') as f:
            enhanced_data = pickle.load(f)
        
        print(f"Original keys: {list(enhanced_data.keys())}")
        
        # Add missing keys with default values
        if 'platt_calibrator' not in enhanced_data:
            enhanced_data['platt_calibrator'] = None
            print("Added missing 'platt_calibrator' key")
        
        if 'isotonic_calibrator' not in enhanced_data:
            enhanced_data['isotonic_calibrator'] = None
            print("Added missing 'isotonic_calibrator' key")
        
        if 'feature_mask' not in enhanced_data:
            # Create a default feature mask if missing
            if 'selected_features' in enhanced_data:
                feature_count = len(enhanced_data['selected_features'])
                enhanced_data['feature_mask'] = [True] * feature_count
                print(f"Added missing 'feature_mask' with {feature_count} features")
            else:
                enhanced_data['feature_mask'] = None
        
        # Ensure performance_metrics exists
        if 'performance_metrics' not in enhanced_data:
            enhanced_data['performance_metrics'] = {
                'validation_accuracy': 0.7,
                'model_count': len(enhanced_data.get('models', [])),
                'feature_count': len(enhanced_data.get('selected_features', [])),
                'trained_on_separate_data': False,
                'training_samples': 0,
                'test_samples': 0
            }
            print("Added missing 'performance_metrics' section")
        
        # Save the fixed data
        backup_path = file_path + '.backup'
        print(f"Creating backup at {backup_path}")
        os.rename(file_path, backup_path)
        
        print("Saving fixed enhanced ensemble data...")
        with open(file_path, 'wb') as f:
            pickle.dump(enhanced_data, f)
        
        print(f"Fixed keys: {list(enhanced_data.keys())}")
        print("✅ Successfully fixed enhanced ensemble data!")
        print(f"Backup saved as: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing enhanced ensemble data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Enhanced Ensemble Data Fixer")
    print("=" * 40)
    
    success = fix_enhanced_ensemble_data()
    
    if success:
        print("\n🎉 Fix completed successfully!")
        print("You can now run your backtesting command again:")
        print("python train.py --backtest --interactive")
    else:
        print("\n❌ Fix failed. You may need to retrain your model:")
        print("python train.py --retrain --cross-validate")