import sqlite3


def get_sqlite_data(dbfile, sql):
    conn = sqlite3.connect(dbfile)
    conn.row_factory = sqlite3.Row
    try:
        cu = conn.cursor()
        try:
            cu.execute(sql)
            data = [dict(r) for r in cu.fetchall()]
        finally:
            cu.close()
    finally:
        conn.close()
    return data
