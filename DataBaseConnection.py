import sqlite3
import datetime


class DataBaseConnection:

    def __init__(self):
        self.conn = sqlite3.connect('database/dataBase_PSIO.db')

        if(self.conn == None):
            print("Connection to database failed")
            exit(-1)

        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def insertEvent(self, REG_PLATE_NO , DESCRIPTION ):

        query_select_employee = "SELECT ID_EMPLOYEE FROM CARS WHERE REG_PLATE_NO = ?"
        self.cursor.execute(query_select_employee, (REG_PLATE_NO,))
        id_employee_row = self.cursor.fetchone()

        if id_employee_row is None:
            raise ValueError(f"Nie znaleziono pracownika dla REG_PLATE_NO: {REG_PLATE_NO}")

        id_employee = id_employee_row[0]

        EVENT_TIME = datetime.datetime.now()
        query = "INSERT INTO EVENTS (ID_EMPLOYEE, EVENT_TIME, DESCRIPTION) VALUES (?,?,?)"

        self.cursor.execute(query, (id_employee, EVENT_TIME, DESCRIPTION))
        self.conn.commit()

    def selectEmployee(self, ID_EMPLOYEE):
        query = "SELECT * FROM EMPLOYEE WHERE ID_EMPLOYEE = ?"

        result = self.cursor.execute(query, (ID_EMPLOYEE,))
        result = result.fetchall()

        if len(result) == 0:
            return False

        return True

    def selectEmployeeCar(self, REG_PLATE_NO):
        query = "SELECT * FROM CARS WHERE REG_PLATE_NO = ?"

        result = self.cursor.execute(query, (REG_PLATE_NO,))
        result = result.fetchall()
        if len(result) == 0:
            return False

        return True


    def checkEmployeePlace(self, Parking_space_number, REG_PLATE_NO):
        query = """
            SELECT 
                E.PARKING_SPACE_NO = ? AS is_correct
            FROM 
                CARS C
            JOIN 
                EMPLOYEES E 
            ON 
                C.ID_EMPLOYEE = E.ID_EMPLOYEE
            WHERE 
                C.REG_PLATE_NO = ?;
            """

        self.cursor.execute(query, (Parking_space_number, REG_PLATE_NO))
        result = self.cursor.fetchone()

        if result is None:
            return None
        return bool(result[0])


    def clearEvents(self):
        query = "DELETE FROM EVENTS"
        self.cursor.execute(query)
        self.conn.commit()

    def displayEvents(self):
        query = "SELECT * FROM EVENTS"
        result = self.cursor.execute(query)

        result = result.fetchall()

        for row in result:
            print(row)

db = DataBaseConnection()

db.insertEvent("ELAGF92", "test")
db.clearEvents()