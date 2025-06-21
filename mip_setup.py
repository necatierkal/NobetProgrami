import pandas as pd
import numpy as np
import gurobipy as gp
import openpyxl
from datetime import datetime, timedelta

def read_inputs(directory):
    xls = pd.ExcelFile(directory)
    return {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

# optimization_df = inputs_dict["optimization_parameters"]
def generate_time_slot_df_from_inputs(optimization_df: pd.DataFrame) -> pd.DataFrame:
    # Extract values
    get_param = lambda name: optimization_df.query("parameter == @name")["value"].iloc[0]

    duration = int(get_param("time_slot_duration_minute"))
    days_str = get_param("days_to_work")  # e.g., "{Pazartesi, Salı, Çarşamba}"
    day_start = get_param("day_start")
    day_end = get_param("day_end")

    # Clean and convert days string to list
    days = [d.strip() for d in days_str.strip("{}").split(",")]

    # Generate time slots
    time_slots = []
    slot_id = 1

    for day_idx, day_ID in enumerate(days, start=1):
        start_dt = datetime.combine(datetime.today(), day_start)
        end_dt = datetime.combine(datetime.today(), day_end)

        current_time = start_dt
        time_slot_no = 1  # Reset for each day

        while current_time + timedelta(minutes=duration) <= end_dt:
            time_slots.append({
                'time_slot_id': (day_ID, current_time.strftime("%H:%M")),
                'day_ID': int(day_ID),
                'day_index': day_idx,
                'time_slot_no': time_slot_no,
                'start_time': current_time.strftime("%H:%M"),
                'end_time': (current_time + timedelta(minutes=duration)).strftime("%H:%M")
            })
            current_time += timedelta(minutes=duration)
            time_slot_no += 1

    return pd.DataFrame(time_slots)


# define input setup class
class InputsSetup:
    def __init__(self, inputs_dict):

        # read problem input

        # Problem Sets

        # Create set of Time Slots
        self.time_slot_duration_minute = int(inputs_dict["optimization_parameters"].query("parameter == 'time_slot_duration_minute'")["value"].iloc[0])
        self.time_slot_df = generate_time_slot_df_from_inputs(inputs_dict["optimization_parameters"])
        self.T_set_of_all_time_slots = self.time_slot_df["time_slot_id"].unique().tolist()

        # Set of cleaners
        self.staff_id_df = inputs_dict["personel_ana_liste"]
        self.set_of_staff = inputs_dict["personel_ana_liste"]["personel_ID"].unique().tolist()

        # temizli_sayisi_gruplar
        self.task_group_requirements_df = inputs_dict["temizlik_sayisi_gruplar"]

        # Set of cleaning tasks
        self.areas_id_df = inputs_dict["tum_alanlar"]
        self.areas_master_list = inputs_dict["tum_alanlar"]["alan_ID"].unique().tolist()
        self.area_requirements_df = pd.concat([inputs_dict["ortak_alan_zamanlar"], inputs_dict["bolum_zamanlar"]], ignore_index=True)


        # Tasks Dictionary and DataFrame (X_a_t)
        tasks = []
        # area =1
        for area in self.areas_master_list:
            filtered_df = self.area_requirements_df[self.area_requirements_df["alan_ID"] == area]
            # _, row = list(filtered_df.iterrows())[0]
            for _, row in filtered_df.iterrows():

                day_ID = row["gun_ID"]
                day = row["gun"]
                duration = int(row["gerekli_sure_dk"])
                duration_slots = duration // self.time_slot_duration_minute

                start = datetime.combine(datetime.today(), row["en_erken_baslama"])
                end = datetime.combine(datetime.today(), row["en_gec_bitis"])

                current_start = start
                while current_start + timedelta(minutes=duration) <= end:
                    matching_slot = self.time_slot_df[
                        (self.time_slot_df["day_ID"] == day_ID) &
                        (self.time_slot_df["start_time"] == current_start.strftime("%H:%M"))
                        ]

                    if not matching_slot.empty:
                        start_index = matching_slot.index[0]
                        occupied_slots = self.time_slot_df.loc[start_index:start_index + duration_slots - 1].time_slot_id.tolist()

                        tasks.append({
                            "task_area_ID": row["alan_ID"],
                            "task_area": row["alan_adi"],
                            "task_start_slot": matching_slot.iloc[0]["time_slot_id"],
                            "task_duration_slots": duration_slots,
                            "task_occupied_time_slots": occupied_slots,
                            "task_duration_min": int(row["gerekli_sure_dk"]),
                            "task_earliest_start_time": (row["en_erken_baslama"]),
                            "task_latest_end_time": (row["en_gec_bitis"]),
                            "task_min_cleaner": int(row["en_az_personel"]),
                            "task_max_cleaner": int(row["en_fazla_personel"]),
                            "temizlik_sayisi_grubu": row.get("temizlik_sayisi_grubu", None)
                        })

                    current_start += timedelta(minutes=self.time_slot_duration_minute)

        self.tasks_dict = tasks
        self.tasks_df = pd.DataFrame(tasks)


        # Preload input data
        df = inputs_dict["personel_alan_uygunlugu"]
        # Set second row as header
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        # Replace the cleaned df back
        inputs_dict["personel_alan_uygunlugu"] = df

        capability_df = inputs_dict["personel_alan_uygunlugu"]
        availability_df = inputs_dict["personel_calisma_saatleri"]



        tasks_staff = []
        # _, task = list(self.tasks_df.iterrows())[0]
        for _, task in self.tasks_df.iterrows():

            task_area_ID = int(task["task_area_ID"])
            task_day_ID = int(task["task_start_slot"][0])  # e.g., 'Pazartesi'
            task_end_time = task["task_latest_end_time"]
            # staff_ID = 1
            for staff_ID in capability_df["personel_ID"]:
                # Check capability
                if capability_df.loc[capability_df["personel_ID"] == staff_ID, task_area_ID].values[0] != 1:
                    continue

                # Check availability
                staff_avail = availability_df[
                    (availability_df["personel_ID"] == staff_ID) &
                    (availability_df["gun_ID"] == task_day_ID)
                    ]

                if staff_avail.empty:
                    continue

                latest_end = staff_avail["calisabilecek_saat_bitis"].values[0]
                if task_end_time > latest_end:
                    continue

                # Staff is eligible for this task
                new_row = task.to_dict()
                new_row["staff_ID"] = staff_ID
                tasks_staff.append(new_row)

        # Final DataFrame
        self.tasks_staff_df = pd.DataFrame(tasks_staff)
        # Reorder columns to place 'staff' after 'task_start_slot'
        cols = list(self.tasks_staff_df.columns)
        if "staff_ID" in cols and "task_start_slot" in cols:
            cols.insert(cols.index("task_start_slot") + 1, cols.pop(cols.index("staff_ID")))
            self.tasks_staff_df = self.tasks_staff_df[cols]


        # Gurobi inputs
        # tasks
        self.tasks_multidict_input = {}  # dictionary of elements of node attribute class
        # Define columns to exclude from value list (used in the key)
        key_cols = ['task_area_ID', 'task_start_slot']
        value_cols = [col for col in self.tasks_df.columns if col not in key_cols]

        # Build multidict-compatible input
        self.tasks_multidict_input = {
            (row['task_area_ID'], row['task_start_slot']): [row[col] for col in value_cols]
            for _, row in self.tasks_df.iterrows()
        }

        (self.task_keys_dict,
         self.task_name_dict,
         self.task_duration_slots_dict,
         self.task_occupied_time_slots_dict,
         self.task_duration_min_dict,
         self.task_earliest_start_time_dict,
         self.task_latest_end_time_dict,
         self.task_min_cleaner_dict,
         self.task_max_cleaner_dict,
         self.task_group_dict) = gp.multidict(self.tasks_multidict_input)



        # task-staff matches
        self.tasks_staff_multidict_input = {}
        # Define key columns and extract value columns
        key_cols_staff = ['task_area_ID', 'task_start_slot', 'staff_ID']
        value_cols_staff = [col for col in self.tasks_staff_df.columns if col not in key_cols_staff]

        # Build multidict input
        self.tasks_staff_multidict_input = {
            (row['task_area_ID'], row['task_start_slot'], row['staff_ID']): [row[col] for col in value_cols_staff]
            for _, row in self.tasks_staff_df.iterrows()
        }

        (self.task_staff_keys_dict,
         self.task_name_dict,
         self.task_duration_slots_staff_dict,
         self.task_occupied_time_slots_staff_dict,
         self.task_duration_min_staff_dict,
         self.task_earliest_start_time_staff_dict,
         self.task_latest_end_time_staff_dict,
         self.task_min_cleaner_staff_dict,
         self.task_max_cleaner_staff_dict,
         self.task_group_staff_dict) = gp.multidict(self.tasks_staff_multidict_input)



        # task-staff-occupation
        self.task_staff_occupation_tuple_list = []
        seen = set()

        for _, row in self.tasks_staff_df.iterrows():
            area = row["task_area_ID"]
            staff = row["staff_ID"]
            for slot in row["task_occupied_time_slots"]:
                key = (area, slot, staff)
                if key not in seen:
                    seen.add(key)
                    self.task_staff_occupation_tuple_list.append(key)