import numpy as np
from flask import Flask, render_template, jsonify, request
from sympy import symbols, sympify, diff
from scipy.integrate import odeint

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/lu-decomposition')
def lu_decomposition():
    return render_template('pages/lu_decomposition.html')

@app.route('/matrix')
def matrix():
    return render_template('pages/matrix.html')

@app.route('/matrix-inverse')
def matrix_inverse():
    return render_template('pages/matrix_inverse.html')

@app.route('/gauss-elimination')
def gauss_elimination():
    return render_template('pages/gauss_elimination.html')

@app.route('/iteration-method')
def iteration_method():
    return render_template('pages/iteration_method.html')

@app.route('/non-linear')
def non_linear():
    return render_template('pages/non_linear.html')

@app.route('/interpolation')
def interpolation():
    return render_template('pages/interpolation.html')

@app.route('/system-simulation')
def system_simulation():
    return render_template('pages/system_simulation.html')

@app.route('/monte-carlo')
def monte_carlo():
    return render_template('pages/monte_carlo.html')

@app.route('/markov-chain')
def markov_chain():
    return render_template('pages/markov_chain.html')

@app.route('/api/lu-decomposition', methods=['POST'])
def calculate_lu():
    try:
        data = request.get_json()
        matrix = np.array(data['matrix'], dtype=float)
        n = len(matrix)
        
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        
        for i in range(n):
            L[i][i] = 1
        
        for i in range(n):
            for k in range(i, n):
                sum = 0
                for j in range(i):
                    sum += (L[i][j] * U[j][k])
                U[i][k] = matrix[i][k] - sum
            
            for k in range(i + 1, n):
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                if U[i][i] == 0:
                    return jsonify({"error": "Matriks tidak dapat didekomposisi (determinan = 0)"}), 400
                L[k][i] = (matrix[k][i] - sum) / U[i][i]
        
        return jsonify({
            "result": {
                "L": L.tolist(),
                "U": U.tolist()
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/matrix-operation', methods=['POST'])
def matrix_operation():
    try:
        data = request.get_json()
        operation = data['operation']
        matrix_a = np.array(data['matrixA'], dtype=float)
        
        if operation == 'transpose':
            result = matrix_a.T
        else:
            matrix_b = np.array(data['matrixB'], dtype=float)
            if operation == 'add':
                result = matrix_a + matrix_b
            elif operation == 'subtract':
                result = matrix_a - matrix_b
            elif operation == 'multiply':
                result = np.matmul(matrix_a, matrix_b)
        
        return jsonify({
            "result": result.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/matrix-inverse', methods=['POST'])
def calculate_matrix_inverse():
    try:
        data = request.get_json()
        matrix = np.array(data['matrix'], dtype=float)
        
        determinant = np.linalg.det(matrix)
        
        if abs(determinant) < 1e-10:  
            return jsonify({"error": "Matriks singular (determinan = 0), tidak memiliki invers"}), 400
        
        inverse = np.linalg.inv(matrix)
        
        return jsonify({
            "inverse": inverse.tolist(),
            "determinant": determinant
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/gauss-elimination', methods=['POST'])
def calculate_gauss_elimination():
    try:
        data = request.get_json()
        matrix = np.array(data['matrix'], dtype=float)
        constants = np.array(data['constants'], dtype=float)
        
        n = len(matrix)
        m = len(matrix[0])
        augmented = np.column_stack((matrix, constants))
        steps = []
        
        steps.append({
            'matrix': matrix.tolist(),
            'constants': constants.tolist()
        })
        
        for i in range(n):

            pivot = augmented[i][i]
            if abs(pivot) < 1e-10:
                return jsonify({"error": "Sistem tidak memiliki solusi unik"}), 400
            
            augmented[i] = augmented[i] / pivot
            
            for j in range(i + 1, n):
                factor = augmented[j][i]
                augmented[j] = augmented[j] - factor * augmented[i]
            
            steps.append({
                'matrix': augmented[:, :m].tolist(),
                'constants': augmented[:, m].tolist()
            })
        
        solution = np.zeros(m)
        for i in range(n-1, -1, -1):
            solution[i] = augmented[i][m]
            for j in range(i+1, m):
                solution[i] = solution[i] - augmented[i][j] * solution[j]
        
        return jsonify({
            'steps': steps,
            'solution': solution.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/iteration-method', methods=['POST'])
def calculate_iteration():
    try:
        data = request.get_json()
        matrix = np.array(data['matrix'], dtype=float)
        constants = np.array(data['constants'], dtype=float)
        initial = np.array(data['initial'], dtype=float)
        max_iter = data['maxIterations']
        tolerance = data['tolerance']
        
        n = len(matrix)
        x = initial.copy()
        iterations = []
        
        for i in range(n):
            if abs(matrix[i][i]) <= sum(abs(matrix[i])) - abs(matrix[i][i]):
                return jsonify({"error": "Matriks tidak diagonal dominan, konvergensi tidak terjamin"}), 400
        
        for iter in range(max_iter):
            x_old = x.copy()
            iterations.append(x_old.tolist())
            
            for i in range(n):
                sum1 = sum(matrix[i][j] * x[j] for j in range(i))
                sum2 = sum(matrix[i][j] * x_old[j] for j in range(i + 1, n))
                x[i] = (constants[i] - sum1 - sum2) / matrix[i][i]
            
            error = np.max(np.abs((x - x_old) / x)) * 100
            if error < tolerance:
                iterations.append(x.tolist())
                return jsonify({
                    'solution': x.tolist(),
                    'iterations': iterations,
                    'error': error,
                    'converged': True,
                    'iterationCount': iter + 1
                })
        
        iterations.append(x.tolist())
        return jsonify({
            'solution': x.tolist(),
            'iterations': iterations,
            'error': error,
            'converged': False,
            'iterationCount': max_iter
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/non-linear', methods=['POST'])
def calculate_non_linear():
    try:
        data = request.get_json()
        equation = data['equation']
        x0 = float(data['initialGuess'])
        max_iter = int(data['maxIterations'])
        tolerance = float(data['tolerance'])
        
        x = symbols('x')
        f = sympify(equation)
        f_prime = diff(f, x) 
        
        f_lambda = lambda val: float(f.subs(x, val))
        f_prime_lambda = lambda val: float(f_prime.subs(x, val))
        
        iterations = []
        x_current = x0
        
        for i in range(max_iter):
            fx = f_lambda(x_current)
            dfx = f_prime_lambda(x_current)
            
            iterations.append({
                'iteration': i,
                'x': x_current,
                'fx': fx,
                'dfx': dfx
            })
            
            if abs(fx) < tolerance:
                return jsonify({
                    'root': x_current,
                    'iterations': iterations,
                    'converged': True,
                    'iterationCount': i + 1,
                    'error': abs(fx)
                })
            
            if abs(dfx) < 1e-10:
                return jsonify({"error": "Turunan mendekati nol, metode tidak dapat dilanjutkan"}), 400
            
            x_next = x_current - fx / dfx
            
            if abs(x_next - x_current) < tolerance:
                iterations.append({
                    'iteration': i + 1,
                    'x': x_next,
                    'fx': f_lambda(x_next),
                    'dfx': f_prime_lambda(x_next)
                })
                return jsonify({
                    'root': x_next,
                    'iterations': iterations,
                    'converged': True,
                    'iterationCount': i + 2,
                    'error': abs(f_lambda(x_next))
                })
            
            x_current = x_next
        
        return jsonify({
            'root': x_current,
            'iterations': iterations,
            'converged': False,
            'iterationCount': max_iter,
            'error': abs(f_lambda(x_current))
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/interpolation', methods=['POST'])
def calculate_interpolation():
    try:
        data = request.get_json()
        x_points = np.array(data['x_points'], dtype=float)
        y_points = np.array(data['y_points'], dtype=float)
        x_interpolate = float(data['x_interpolate'])
        method = data['method']
        
        if method == 'linear':
            result = np.interp(x_interpolate, x_points, y_points)
            
            x_plot = np.linspace(min(x_points), max(x_points), 100)
            y_plot = np.interp(x_plot, x_points, y_points)
            
        elif method == 'polynomial':
            coefficients = np.polyfit(x_points, y_points, len(x_points)-1)
            result = np.polyval(coefficients, x_interpolate)
            
            x_plot = np.linspace(min(x_points), max(x_points), 100)
            y_plot = np.polyval(coefficients, x_plot)
            
        elif method == 'lagrange':
            result = 0
            for i in range(len(x_points)):
                term = y_points[i]
                for j in range(len(x_points)):
                    if i != j:
                        term = term * (x_interpolate - x_points[j])/(x_points[i] - x_points[j])
                result += term
            
            x_plot = np.linspace(min(x_points), max(x_points), 100)
            y_plot = []
            for x in x_plot:
                y = 0
                for i in range(len(x_points)):
                    term = y_points[i]
                    for j in range(len(x_points)):
                        if i != j:
                            term = term * (x - x_points[j])/(x_points[i] - x_points[j])
                    y += term
                y_plot.append(y)
            
        return jsonify({
            'result': float(result),
            'plot_data': {
                'x_points': x_points.tolist(),
                'y_points': y_points.tolist(),
                'x_plot': x_plot.tolist(),
                'y_plot': y_plot if isinstance(y_plot, list) else y_plot.tolist(),
                'x_interpolate': x_interpolate,
                'y_interpolate': float(result)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/system-simulation', methods=['POST'])
def calculate_system_simulation():
    try:
        data = request.get_json()
        
        alpha = float(data['parameters']['alpha'])
        beta = float(data['parameters']['beta'])
        gamma = float(data['parameters']['gamma'])
        delta = float(data['parameters']['delta'])
        
        initial_prey, initial_predator = data['initialConditions']
        
        t_start = float(data['timePoints']['start'])
        t_end = float(data['timePoints']['end'])
        steps = int(data['timePoints']['steps'])
        
        t = np.linspace(t_start, t_end, steps)
        
        def predator_prey(state, t, alpha, beta, gamma, delta):
            prey, predator = state
            dprey_dt = alpha * prey - beta * prey * predator
            dpredator_dt = delta * prey * predator - gamma * predator
            return [dprey_dt, dpredator_dt]
        
        initial_conditions = [initial_prey, initial_predator]
        
        solution = odeint(
            predator_prey, 
            initial_conditions, 
            t, 
            args=(alpha, beta, gamma, delta)
        )
        
        prey = solution[:, 0]
        predator = solution[:, 1]
        
        return jsonify({
            'time': t.tolist(),
            'prey': prey.tolist(),
            'predator': predator.tolist()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/monte-carlo', methods=['POST'])
def calculate_monte_carlo():
    try:
        data = request.get_json()
        num_points = int(data['numPoints'])
        x_min = float(data['xMin'])
        x_max = float(data['xMax'])
        equation = data['equation']
        
        x = symbols('x')
        f = sympify(equation)
        f_lambda = lambda val: float(f.subs(x, val))
        
        x_samples = np.linspace(x_min, x_max, 1000)
        y_samples = [f_lambda(x) for x in x_samples]
        y_min = min(y_samples)
        y_max = max(y_samples)
        
        y_range = max(abs(y_min), abs(y_max))
        y_lower = -y_range if y_min < 0 else 0
        y_upper = y_range
        
        x_points = np.random.uniform(x_min, x_max, num_points)
        y_points = np.random.uniform(y_lower, y_upper, num_points)
        
        x_plot = np.linspace(x_min, x_max, 200)
        y_plot = [f_lambda(x) for x in x_plot]
        

        points_under_positive = []  
        points_over_positive = []   
        points_under_negative = []  
        points_over_negative = []   
        
        points_contributing = 0 
        
        for i in range(num_points):
            x_val = x_points[i]
            y_val = y_points[i]
            y_curve = f_lambda(x_val)
            point = {'x': float(x_val), 'y': float(y_val)}
            
            if y_curve >= 0: 
                if 0 <= y_val <= y_curve:
                    points_contributing += 1
                    points_under_positive.append(point)
                elif 0 <= y_val <= y_upper:
                    points_over_positive.append(point)
            else: 
                if y_curve <= y_val <= 0:
                    points_contributing += 1
                    points_under_negative.append(point)
                elif y_lower <= y_val <= 0:
                    points_over_negative.append(point)
        
        total_box_area = (x_max - x_min) * (y_upper - y_lower)
        estimated_area = total_box_area * (points_contributing / num_points)
        
        from scipy.integrate import quad
        exact_area, _ = quad(lambda x: abs(f_lambda(x)), x_min, x_max)
        
        error_percentage = abs((estimated_area - exact_area) / exact_area) * 100
        
        return jsonify({
            'estimated_area': float(estimated_area),
            'exact_area': float(exact_area),
            'error_percentage': float(error_percentage),
            'pointsContributing': points_contributing,
            'totalPoints': num_points,
            'ratio': points_contributing / num_points,
            'plotData': {
                'xPlot': x_plot.tolist(),
                'yPlot': y_plot,
                'points': {
                    'underPositive': points_under_positive,
                    'overPositive': points_over_positive,
                    'underNegative': points_under_negative,
                    'overNegative': points_over_negative
                },
                'yRange': {
                    'min': float(y_lower),
                    'max': float(y_upper)
                }
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/markov-chain', methods=['POST'])
def calculate_markov_chain():
    try:
        data = request.get_json()
        transition_matrix = np.array(data['transitionMatrix'], dtype=float)
        initial_state = np.array(data['initialState'], dtype=float)
        num_steps = int(data['numSteps'])
        
        row_sums = np.sum(transition_matrix, axis=1)
        if not np.allclose(row_sums, 1.0):
            return jsonify({"error": "Jumlah probabilitas setiap baris harus 1"}), 400
        
        if not np.all((transition_matrix >= 0) & (transition_matrix <= 1)):
            return jsonify({"error": "Probabilitas harus antara 0 dan 1"}), 400
        
        current_state = initial_state
        states = [current_state.tolist()]
        
        for _ in range(num_steps):
            current_state = np.dot(current_state, transition_matrix)
            states.append(current_state.tolist())
        
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        steady_state_idx = np.argmin(np.abs(eigenvalues - 1.0))
        steady_state = eigenvectors[:, steady_state_idx].real
        steady_state = steady_state / np.sum(steady_state) 
        
        num_states = len(initial_state)
        state_labels = [f"State {i+1}" for i in range(num_states)]
        
        return jsonify({
            'states': states,
            'steadyState': steady_state.tolist(),
            'stateLabels': state_labels,
            'numSteps': num_steps
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
