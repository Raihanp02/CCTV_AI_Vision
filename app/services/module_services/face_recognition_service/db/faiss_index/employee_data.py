import pickle
import threading

class EmployeeData:
    def __init__(self, path="id_map.pkl"):
        self.load_id_map(path)
        self.lock = threading.RLock()

    def load_id_map(self, path="id_map.pkl"):
        with open(path, "rb") as f:
            self.employee_repository = pickle.load(f)

    def save_id_map(self, path="id_map.pkl"):
        with self.lock:
            with open(path, "wb") as f:
                pickle.dump(self.employee_repository, f)

    def add_employee(self, employee_id, employee_data):
        with self.lock:
            if not self.employee_repository.get(employee_id, False):
                print("employee id already exist delete it if want to change")
                return
            self.employee_repository[employee_id] = employee_data
            self.save_id_map()

    def delete_employee(self, employee_id):
        if self.employee_repository.get(employee_id, False):
            del self.employee_repository[employee_id]
        
        else:
            print("employee id doesnt exist")
            return 

    def get_employee(self, employee_id):
        try:
            employee = self.employee_repository.get(employee_id)
            return employee
        except Exception:
            raise ValueError(f"Employee with ID {employee_id} not found.")
        
    def get_name(self, employee_id):
        try:
            employee = self.get_employee(employee_id)
            return employee.get('name', 'Unknown')
        except ValueError:
            raise ValueError(f"Employee ID {employee_id} name not found.")