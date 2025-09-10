• Trained and evaluated Logistic Regression, Random Forest, XGBoost on a 255K loans dataset; scaled 7 continuous features, encoded 9 categorical; handled 7:1 imbalance via resampling and class_weight/scale_pos_weight; F1.5-based thresholding and probability calibration.
• Built MLP in PyTorch optimized with LibAUC PESG (AUCM/AP losses), achieving AUC of 0.75 and AP loss of 0.300.
• Achieved AUC of 0.74 for Logistic Regression, 0.72 for RF/XGB; recall of 0.65 and precision of 0.26 at the F1.5 optimized threshold.
• Summarized findings in detailed report



This repository contains:
1. Preprocessing code for all 16 variables
2. Full source code for Logistic Regression, Random Forest, XGBoost, and MLP models
3. Final report summarizing the results for Logistic Regression, Random Forest, and XGBoost models.

Feel free to reach out with any questions or issues!
