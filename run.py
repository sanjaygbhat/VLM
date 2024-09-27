from app import create_app, db
from app.models.user import User  # Import your models here
import os
from sqlalchemy.exc import OperationalError

app = create_app()

def init_db():
    with app.app_context():
        try:
            # Try to query the database
            User.query.first()
        except OperationalError:
            # If the query fails, it likely means the tables don't exist
            print("Database tables not found. Creating tables...")
            db.create_all()
            print("Database tables created successfully.")
        else:
            print("Database tables already exist. Skipping creation.")

if __name__ == '__main__':
    # Check if the database file exists
    db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}. Initializing database...")
        init_db()
    else:
        print(f"Database file found at {db_path}.")
        # Even if the file exists, we'll check if the tables are created
        init_db()

    app.run(debug=True, host='0.0.0.0', port=5000)