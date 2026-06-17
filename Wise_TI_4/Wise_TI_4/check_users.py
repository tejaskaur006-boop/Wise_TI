# check_users.py
# Run this from inside your Wise_TI backend folder (same place as main.py)

import sqlite3

# Adjust this if your database file has a different name
DB_PATH = "eventflow.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=== All rows in 'users' table ===\n")
cursor.execute("SELECT id, email, role, reference_id, is_active, created_at FROM users")
rows = cursor.fetchall()

if not rows:
    print("No users found in the table at all!")
else:
    for row in rows:
        print(f"id={row[0]} | email={row[1]} | role={row[2]} | reference_id={row[3]} | is_active={row[4]} | created_at={row[5]}")

print("\n=== Looking specifically for Karan (participant id 5) ===\n")
cursor.execute("SELECT * FROM users WHERE reference_id = 5 AND role = 'PARTICIPANT'")
karan_user = cursor.fetchone()

if karan_user:
    print("Found a user row for Karan:")
    print(karan_user)
else:
    print("No matching user row found for participant id 5 (Karan). This means his login/password was never created!")

conn.close()