import sqlite3

# Initialize the SQLite database and create the table
def init_db():
    conn = sqlite3.connect('model_classes.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS class_names (
        model_name TEXT,
        class_index INTEGER,
        class_name TEXT
    )
    ''')
    conn.commit()
    conn.close()

# Insert class names for the given model
def insert_class_names(model_name: str, class_names: list):
    conn = sqlite3.connect('model_classes.db')
    cursor = conn.cursor()

    # Insert class names into the database
    for index, class_name in enumerate(class_names):
        cursor.execute('INSERT INTO class_names (model_name, class_index, class_name) VALUES (?, ?, ?)', (model_name, index, class_name.strip()))
    
    conn.commit()
    conn.close()

# Retrieve class name by model name and index
def get_class_name(model_name: str, class_index: int):
    print(model_name, class_index)
    conn = sqlite3.connect('model_classes.db')
    cursor = conn.cursor()
    cursor.execute('SELECT class_name FROM class_names WHERE model_name = ? AND class_index = ?', (model_name, int(class_index)))
    result = cursor.fetchone()
    conn.close()
    return result[0]