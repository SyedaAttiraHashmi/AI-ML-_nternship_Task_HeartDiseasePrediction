
# Save the trained Logistic Regression pipeline
import joblib

joblib.dump(logreg_pipeline, 'heart_disease_model.pkl')
print("Model saved as heart_disease_model.pkl")
