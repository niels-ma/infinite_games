"""Manual script to update events in the database for manual tests in codespaces."""

import random
import sqlite3

if __name__ == "__main__":
    # Connect to your SQLite database
    conn = sqlite3.connect("/workspace/validator_test.db")
    cursor = conn.cursor()

    # Retrieve the rowids from the predictions table
    cursor.execute("SELECT rowid FROM events")
    rows = cursor.fetchall()
    all_ids = [row[0] for row in rows]

    # Calculate number of records to update (20% of total)
    num_to_update = int(len(all_ids) * 0.2)

    # Randomly select 20% of rowids
    random_ids = random.sample(all_ids, num_to_update) if num_to_update > 0 else []

    # Update selected records: set status = 3 and outcome = 1
    if random_ids:
        placeholders = ",".join("?" for _ in random_ids)
        update_query = f"""
            UPDATE events
            SET status = 3,
            outcome = 1,
            resolved_at = CURRENT_TIMESTAMP
            WHERE rowid IN ({placeholders})
        """
        cursor.execute(update_query, random_ids)
        conn.commit()

    conn.close()
