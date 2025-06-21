import pandas as pd
import gurobipy as gp
import re
from gurobipy import GRB
from collections import OrderedDict
import ast
import seaborn as sns
import matplotlib.pyplot as plt




# organize results
def model_organize_results(var_values, var_df):
    counter = 0
    for v in var_values:
        var_name = v.varName
        base, index_str = var_name.split("[", 1)
        index_str = index_str.rstrip("]")  # Remove closing bracket
        # Convert the index string into an actual tuple
        key_tuple = ast.literal_eval(f"({index_str})")  # forces it to tuple even if 2 elements
        # Make each element a string
        current_var = [base] + [str(x) for x in key_tuple]
        current_var.append(round(v.X, 2))
        var_df.loc[counter] = current_var
        counter = counter + 1
        # with open("./math_model_outputs/" + 'mip-results.txt',
        #           "w") as f:  # a: open for writing, appending to the end of the file if it exists
        #     f.write(','.join(map(str, current_var)) + '\n')
        # print(','.join(map(str,current_var )))
    return var_df

day_map = {
    '1': 'Monday',
    '2': 'Tuesday',
    '3': 'Wednesday',
    '4': 'Thursday',
    '5': 'Friday',
    1: 'Monday',
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
}
def mathematical_model_solve(mip_inputs):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from openpyxl import load_workbook
    from openpyxl.styles import Border, Side
    import pandas as pd

    model = gp.Model("hu_cleaning_staff_scheduling")

    # add x_at variables - if the task a is assigned to start at time slot t
    x_at = model.addVars(
        mip_inputs.task_keys_dict,
        vtype=GRB.BINARY,
        name="x_at",
    )


    x_atj = model.addVars(
        mip_inputs.task_staff_keys_dict,
        vtype=GRB.BINARY,
        name="x_atj"
    )


    y_atj = model.addVars(
        mip_inputs.task_staff_occupation_tuple_list,
        vtype=GRB.BINARY,
        name="y_atj"
    )


    w_j = model.addVars(
        mip_inputs.set_of_staff,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="w_j",
    )

    w_max = model.addVar(
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="w_max",
    )

    for var in model.getVars():
        try:
            print(var.getAttr("VarName"))
        except UnicodeDecodeError:
            print("[Error decoding variable name]")


    # set objective (1)
    alpha_coefficient = 0.01 # Coefficient that determines the reward for assigning more than the minimum number of cleaners.
    obj_min = w_max - alpha_coefficient * alpha_coefficient * gp.quicksum(x_atj.values())
    model.setObjective(obj_min)

    # constraint 2 - ensures that each area a is assigned to exactly one starting time slot.
    # a=3
    for a in mip_inputs.areas_master_list:
        cleaning_number_groups = mip_inputs.tasks_df.query("task_area_ID == @a")["temizlik_sayisi_grubu"].unique().tolist()
        # g=cleaning_number_groups[0]
        for g in cleaning_number_groups:
            matching_keys = [
                key for key, value in mip_inputs.task_group_dict.items()
                if key[0] == a and (
                    pd.isna(value) if pd.isna(g) else value == g
                )
            ]
            if pd.isna(g):
                model.addConstr(gp.quicksum(x_at[key] for key in matching_keys) == 1)
            else:
                required_number = mip_inputs.task_group_requirements_df.query("temizlik_sayisi_grubu == @g")["temizlik_sayisi"].iloc[0]
                model.addConstr(gp.quicksum(x_at[key] for key in matching_keys) == required_number)

    # constraint 3 - ensures that cleaning tasks start only at feasible times.
    # we handle this when generating variables. Non feasible times are not available as a variable.


    # Constraint 4 and 5-
    # 4 ensures that the minimum required number of cleaners is assigned to an area for the time slot the
    # 5 nsures that the maximum number of cleaners assigned to an area during a time
    # a=1
    for a in mip_inputs.areas_master_list:
        starting_time_slots_of_taks_a =  mip_inputs.tasks_df.query("task_area_ID == @a")["task_start_slot"].unique().tolist()
        # t= starting_time_slots_of_taks_a[0]
        for t in starting_time_slots_of_taks_a:

           model.addConstr(gp.quicksum(var for (a_0, t_0, j_0), var in x_atj.items() if a_0 == a and t_0 == t) >= mip_inputs.task_min_cleaner_dict[a,t] * x_at[(a, t)])
           model.addConstr(gp.quicksum(var for (a_0, t_0, j_0), var in x_atj.items() if a_0 == a and t_0 == t) <= mip_inputs.task_max_cleaner_dict[a, t] * x_at[(a, t)])

    # Constraint 6 once a cleaning task starts (a cleaner is assigned to area a at time t to start, the cleaner will continue cleaning the area for the entire required duration for that area.
    # key = list(x_atj.keys())[0]
    for key in x_atj.keys():
        slots_temp = mip_inputs.task_occupied_time_slots_staff_dict[key]
        model.addConstr(
            gp.quicksum(var for ((a_0, t_0, j_0), var) in y_atj.items()
            if a_0 == key[0] and t_0 in slots_temp and j_0 == key[2] ) == mip_inputs.task_duration_slots_staff_dict[key]*x_atj[key])


    # Constraint 7 ensures that each cleaner is assigned to only one area per time slot. This prevents a cleaner from being assigned to multiple areas at the same time.
    # j=1
    for j in mip_inputs.set_of_staff:
        query_result = mip_inputs.tasks_staff_df.query("staff_ID == @j")["task_occupied_time_slots"]
        # Flatten and preserve order of first occurrence
        set_of_times_that_staff_j_can_work = list(OrderedDict.fromkeys(
            slot for lst in query_result for slot in lst
        ))
        # t= set_of_times_that_staff_j_can_work[0]
        for t in set_of_times_that_staff_j_can_work:
            model.addConstr(
                gp.quicksum(
                    var for (a_, t_, j_), var in y_atj.items()
                    if t_ == t and j_ == j
                ) <= 1
            )

        # Constraint 9 calculates the total workload wj for each cleaner j.
        # j=1
        for j in mip_inputs.set_of_staff:
            model.addConstr(w_j[j] == y_atj.sum('*','*',j))


        # Constraint 10
        for j in mip_inputs.set_of_staff:
            model.addConstr(w_j[j] <= w_max )




    model.update()
    # model.write("model_hand.lp")
    # model.printStats()
    model.optimize()

    if model.Status == GRB.Status.INFEASIBLE:
        model.computeIIS()
        model.write("infeasible_model.ilp")
        print("There is no feasible solution for the given inputs. Go check infeasible_model.ilp file for more details.")

    else:
        x_at_results_df = pd.DataFrame(columns=['var_name', 'area', 'time_slot_to_start', 'value'])
        x_at_results_df = model_organize_results([v for v in x_at.values() if v.X > 0], x_at_results_df)
        x_at_results_df[['day', 'time']] = x_at_results_df['time_slot_to_start'].apply(
            lambda s: pd.Series(ast.literal_eval(s))
        )
        cols = [col for col in x_at_results_df.columns if col != 'value'] + ['value']
        x_at_results_df = x_at_results_df[cols]
        x_at_results_df['day'] = x_at_results_df['day'].replace(day_map)

        x_atj_results_df = pd.DataFrame(columns=['var_name', 'area', 'time_slot_to_start', 'staff', 'value'])
        x_atj_results_df = model_organize_results([v for v in x_atj.values() if v.X > 0], x_atj_results_df)

        y_atj_results_df = pd.DataFrame(columns=['var_name', 'area', 'time_slot_occupied', 'staff', 'value'])
        y_atj_results_df = model_organize_results([v for v in y_atj.values() if v.X > 0], y_atj_results_df)
        y_atj_results_df[['day', 'time']] = y_atj_results_df['time_slot_occupied'].apply(
            lambda s: pd.Series(ast.literal_eval(s))
        )
        cols = [col for col in y_atj_results_df.columns if col != 'value'] + ['value']
        y_atj_results_df = y_atj_results_df[cols]
        y_atj_results_df['day'] = y_atj_results_df['day'].replace(day_map)

       # w_j_results_df = pd.DataFrame(columns=['var_name', 'staff', 'workload'])
        # w_j_results_df = model_organize_results([v for v in w_j.values() if v.X > 0], w_j_results_df)

        data = []
        for v in w_j.values():
            var_name, index_str = v.varName.split("[", 1)  # e.g., "w_j", "1]"
            staff = int(index_str.rstrip("]"))  # clean closing bracket and convert to int
            workload = round(v.X, 2)
            data.append([var_name, staff, workload])

        w_j_results_df = pd.DataFrame(data, columns=["var_name", "staff", "workload"])




        global_results_df = pd.DataFrame(
            columns=['max_workload', 'model_obj_value', 'model_obj_bound', 'gap', 'gurobi_time'])
        global_results_df.loc[len(global_results_df.index)] = [w_max.X, model.objval, model.objbound, model.mipgap,
                                                               model.runtime]

        writer = pd.ExcelWriter('outputs/results.xlsx', engine='openpyxl')
        global_results_df.to_excel(writer, sheet_name='global_results')
        x_at_results_df.to_excel(writer, sheet_name='x_at_results_df')
        x_atj_results_df.to_excel(writer, sheet_name='x_atj_results_df')
        y_atj_results_df.to_excel(writer, sheet_name='y_atj_results_df')
        w_j_results_df.to_excel(writer, sheet_name='w_j_results_df')

        writer.close()








        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Copy original DataFrame
        df = y_atj_results_df.copy()

        # Ensure weekday order
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        df['day'] = pd.Categorical(df['day'], categories=weekday_order, ordered=True)

        # Create time slot label: e.g., "Monday 07:00"
        df['time_label'] = df['day'].astype(str) + ' ' + df['time']
        df['time_label'] = pd.Categorical(
            df['time_label'],
            categories=sorted(df['time_label'].unique(),
                              key=lambda x: (weekday_order.index(x.split()[0]), x.split()[1])),
            ordered=True
        )

        # Ensure area and staff are integers
        df['area'] = df['area'].astype(int)
        df['staff'] = df['staff'].astype(int)

        # Pivot: index = area (Y axis), columns = time (X axis), values = staff
        matrix = df.pivot_table(
            index='area',
            columns='time_label',
            values='staff',
            aggfunc=lambda x: ', '.join(map(str, sorted(set(x)))),
        fill_value=''
        )

        # Sort area from 1 to 17
        matrix = matrix.sort_index(ascending=True)

        # Plot
        plt.figure(figsize=(16, 8))
        sns.heatmap(
            matrix != '',  # mask: where staff is assigned
            cmap="YlGnBu",
            cbar=False,
            annot=matrix,
            fmt='',
            annot_kws={"size": 7},  # ← smaller font for staff IDs
            linewidths=0.5,
            linecolor='gray'
        )
        plt.title("Staff Assigned to Each Area Over Time")
        plt.xlabel("Time Slot")
        plt.ylabel("Area")
        plt.xticks(rotation=90, ha='center')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Export the matrix to Excel
        id_to_name = dict(zip(mip_inputs.staff_id_df['personel_ID'], mip_inputs.staff_id_df['personel']))

        def replace_ids_with_names(cell):
            if not isinstance(cell, str) or cell.strip() == '':
                return ''
            staff_ids = [int(x.strip()) for x in cell.split(',')]
            staff_names = [id_to_name.get(x, f"Unknown({x})") for x in staff_ids]
            return ', '.join(staff_names)

        # Replace staff IDs with names
        matrix_named = matrix.apply(lambda col: col.map(replace_ids_with_names))

        # ✅ Replace area IDs (index) with area names
        area_id_to_name = dict(zip(mip_inputs.areas_id_df['alan_ID'], mip_inputs.areas_id_df['alan_adi']))
        matrix_named = matrix_named.rename(index=area_id_to_name)

        # Copy and prepare multi-level columns (Day, Time)
        matrix_named_copy = matrix_named.copy()
        matrix_named_copy.columns = pd.MultiIndex.from_tuples(
            [tuple(col.split(' ', 1)) for col in matrix_named_copy.columns],
            names=['Day', 'Time']
        )

        # Sort columns by weekday then time
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        matrix_named_copy = matrix_named_copy.reindex(
            columns=sorted(matrix_named_copy.columns, key=lambda x: (weekday_order.index(x[0]), x[1]))
        )

        # Write the DataFrame to Excel
        excel_path = 'outputs/results.xlsx'
        sheet_name = 'alan-personel-takvim'

        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            matrix_named_copy.to_excel(writer, sheet_name=sheet_name)







    # staff based plot
    # Create working copy
    df = y_atj_results_df.copy()

    # Ensure English weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df['day'] = pd.Categorical(df['day'], categories=weekday_order, ordered=True)

    # Create combined time label: e.g., "Monday 07:00"
    df['time_label'] = df['day'].astype(str) + ' ' + df['time']

    # Sort time labels: by day, then time
    df['time_label'] = pd.Categorical(
        df['time_label'],
        categories=sorted(df['time_label'].unique(),
                          key=lambda x: (weekday_order.index(x.split()[0]), x.split()[1])),
        ordered=True
    )

    # ✅ Ensure staff is int to sort correctly
    df['staff'] = df['staff'].astype(int)

    # Pivot: staff vs time, values = area
    matrix = df.pivot(index='staff', columns='time_label', values='area')

    # Sort staff from 1 at top to max at bottom
    matrix = matrix.sort_index(ascending=True)

    # Build annotation matrix (replace NaNs with "")
    annot_matrix = matrix.fillna("").astype(str)

    # Plot heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        matrix.notnull(),  # Show presence of assignments
        cmap="YlGnBu",
        cbar=False,
        annot=annot_matrix,  # Show area ID or blank
        fmt='',
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("Which Area Was Cleaned by Each Staff Over Time")
    plt.xlabel("Time Slot")
    plt.ylabel("Staff")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Step 1: Setup
    df = y_atj_results_df.copy()

    # Ensure weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df['day'] = pd.Categorical(df['day'], categories=weekday_order, ordered=True)

    # Create combined time label
    df['time_label'] = df['day'].astype(str) + ' ' + df['time']
    df['time_label'] = pd.Categorical(
        df['time_label'],
        categories=sorted(df['time_label'].unique(), key=lambda x: (weekday_order.index(x.split()[0]), x.split()[1])),
        ordered=True
    )

    # Convert staff and area to int
    df['staff'] = df['staff'].astype(int)
    df['area'] = df['area'].astype(int)

    # Step 2: Pivot: staff (rows), time (columns), values = area ID
    matrix = df.pivot(index='staff', columns='time_label', values='area')

    # Step 3: Replace area IDs with names (cell values)
    area_id_to_name = dict(zip(mip_inputs.areas_id_df['alan_ID'], mip_inputs.areas_id_df['alan_adi']))
    matrix_named = matrix.applymap(lambda x: area_id_to_name.get(x, '') if pd.notna(x) else '')

    # Step 4: Replace staff ID index with names
    staff_id_to_name = dict(zip(mip_inputs.staff_id_df['personel_ID'], mip_inputs.staff_id_df['personel']))
    matrix_named.index = matrix_named.index.map(staff_id_to_name)

    # Step 5: Multi-level columns (Day, Time)
    matrix_named.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split(' ', 1)) for col in matrix_named.columns],
        names=['Day', 'Time']
    )

    # Sort columns by day, then time
    matrix_named = matrix_named.reindex(
        columns=sorted(matrix_named.columns, key=lambda x: (weekday_order.index(x[0]), x[1]))
    )

    # Step 6: Export to Excel
    excel_path = 'outputs/results.xlsx'
    sheet_name = 'personel-alan-takvim'

    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        matrix_named.to_excel(writer, sheet_name=sheet_name)

    print(f"✅ Exported staff-based area schedule to sheet '{sheet_name}' in {excel_path}")








    # ✅ Summary Plot 1: Number of Tasks per Staff

    import seaborn as sns
    import matplotlib.pyplot as plt

    staff_counts = y_atj_results_df['staff'].value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    sns.barplot(x=staff_counts.index, y=staff_counts.values, palette='Blues_d')
    plt.title('Number of Cleaning Tasks per Staff')
    plt.xlabel('Staff ID')
    plt.ylabel('Number of Tasks')
    plt.tight_layout()
    plt.show()


    #  1. Calendar Heatmap (Day × Time)
    # Goal: Show how busy each time slot is across the week.
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = y_atj_results_df.copy()

    # Count number of tasks per (day, time) — reflects busy staff
    calendar = df.groupby(['day', 'time']).size().reset_index(name='busy_staff_count')

    # Pivot for heatmap
    heatmap_staff = calendar.pivot(index='day', columns='time', values='busy_staff_count').fillna(0)

    # Ensure correct weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    heatmap_staff = heatmap_staff.reindex(weekday_order)

    plt.figure(figsize=(14, 5))
    sns.heatmap(heatmap_staff, annot=True, fmt='.0f', cmap='YlOrBr', linewidths=0.5)
    plt.title('Calendar Heatmap: Number of Busy Staff per Time Slot')
    plt.xlabel('Time')
    plt.ylabel('Day')
    plt.tight_layout()
    plt.show()

    # Count number of distinct areas cleaned per (day, time)
    area_calendar = df.groupby(['day', 'time'])['area'].nunique().reset_index(name='areas_cleaned_count')

    # Pivot for heatmap
    heatmap_areas = area_calendar.pivot(index='day', columns='time', values='areas_cleaned_count').fillna(0)
    heatmap_areas = heatmap_areas.reindex(weekday_order)

    plt.figure(figsize=(14, 5))
    sns.heatmap(heatmap_areas, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
    plt.title('Calendar Heatmap: Number of Distinct Areas Cleaned per Time Slot')
    plt.xlabel('Time')
    plt.ylabel('Day')
    plt.tight_layout()
    plt.show()








    #
    #
    #
    # for day in weekday_order:
    #     matrix_day = matrix[[col for col in matrix.columns if col.startswith(day)]]
    #     plt.figure(figsize=(14, 6))
    #     sns.heatmap(
    #         matrix_day != '',
    #         cmap="YlGnBu",
    #         cbar=False,
    #         annot=matrix_day,
    #         fmt='',
    #         annot_kws={"size": 10},
    #         linewidths=0.5,
    #         linecolor='gray'
    #     )
    #     plt.title(f"Staff Assigned per Area - {day}")
    #     plt.xlabel("Time")
    #     plt.ylabel("Area")
    #     plt.xticks(rotation=0)
    #     plt.yticks(rotation=0)
    #     plt.tight_layout()
    #     plt.show()
