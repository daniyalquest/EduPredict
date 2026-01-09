import json
import bcrypt
from pymongo import MongoClient

def sync_database():
    # 1. Cloud Connection String
    uri = "mongodb+srv://daniyal2472_db_user:Nc1mhy0mORO9RTLY@cluster-edupredict.wggn6ki.mongodb.net/?appName=Cluster-EduPredict"
    
    try:
        client = MongoClient(uri)
        db = client['edu_predict_db']
        
        # Collections
        records_col = db['student_records']
        users_col = db['users']

        # --- STEP 1: CLEAR EXISTING DATA ---
        print("Connecting to MongoDB Atlas...")
        records_col.delete_many({})
        users_col.delete_many({})  # Clear old users too
        print("✓ Database Cleaned.")

        # --- STEP 2: INSERT STUDENT RECORDS ---
        with open('cleaned.json', 'r') as f:
            student_data = json.load(f)

        if student_data:
            records_col.insert_many(student_data)
            print(f"✓ Pushed {len(student_data)} student records.")

        # --- STEP 3: CREATE AUTHENTIC DEMO USER ---
        demo_username = "admin_demo"
        demo_password = "password123"
        
        # Hashing the password for security
        hashed_pw = bcrypt.hashpw(demo_password.encode('utf-8'), bcrypt.gensalt())
        
        demo_user = {
            "username": demo_username,
            "password": hashed_pw,
            "role": "admin",
            "full_name": "Demo Administrator"
        }
        
        users_col.insert_one(demo_user)
        
        print("\n" + "="*30)
        print("DEMO USER CREATED SUCCESSFULLY")
        print(f"Username: {demo_username}")
        print(f"Password: {demo_password}")
        print("="*30)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sync_database()