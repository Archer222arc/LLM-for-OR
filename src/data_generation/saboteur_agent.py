"""
Saboteur Agent for controlled error injection into optimization models.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A3_Data_Generation.md

Key Components:
    - SaboteurAgent: Main class for error injection
    - Six error types:
        - A (bound): Flip constraint direction
        - B (variable): Change variable type
        - C (logic): Remove/modify coefficients
        - D (conflict): Add contradicting constraint
        - E (multi-constraint): Requires 2+ fixes simultaneously
        - F (hidden dependency): Root cause not directly in IIS

Example:
    >>> from src.solvers import GurobiSolver
    >>> from src.data_generation import SaboteurAgent
    >>> solver = GurobiSolver.from_file("model.mps")
    >>> saboteur = SaboteurAgent(solver)
    >>> result = saboteur.inject_type_a()
    >>> print(f"Injection success: {result.success}")
"""

import random
from typing import Optional, List

import gurobipy as gp

from src.solvers import GurobiSolver
from .error_types import ErrorType, InjectionResult, Difficulty


class SaboteurAgent:
    """
    Injects controlled errors into feasible optimization models.

    The Saboteur Agent transforms feasible models into infeasible/unbounded
    ones by injecting one of four error types:
    - Type A: Flip constraint direction (≤ ↔ ≥)
    - Type B: Change variable type (INTEGER ↔ CONTINUOUS)
    - Type C: Remove terms from constraint expressions
    - Type D: Add contradicting constraints

    Attributes:
        _solver: The GurobiSolver instance to modify
        _model: Direct reference to the underlying Gurobi model
        _injection_history: List of all injections performed
    """

    def __init__(self, solver: GurobiSolver, seed: Optional[int] = None):
        """
        Initialize SaboteurAgent.

        Args:
            solver: GurobiSolver instance with a feasible model
            seed: Random seed for reproducibility
        """
        self._solver = solver
        self._model = solver.model
        self._injection_history: List[InjectionResult] = []

        if seed is not None:
            random.seed(seed)

    # =========================================================================
    # Core Injection Methods
    # =========================================================================

    def inject_error(self, error_type: str) -> InjectionResult:
        """
        Inject a specific type of error.

        Args:
            error_type: One of "A", "B", "C", "D", "E", "F"

        Returns:
            InjectionResult with details of the injection
        """
        type_map = {
            "A": self.inject_type_a,
            "B": self.inject_type_b,
            "C": self.inject_type_c,
            "D": self.inject_type_d,
            "E": self.inject_type_e,
            "F": self.inject_type_f,
        }

        if error_type not in type_map:
            raise ValueError(f"Unknown error type: {error_type}. Must be A, B, C, D, E, or F")

        return type_map[error_type]()

    def inject_random_error(self, include_hard: bool = False) -> InjectionResult:
        """
        Inject a random error type.

        Args:
            include_hard: If True, include Type E and F (harder problems)

        Returns:
            InjectionResult with details of the injection
        """
        if include_hard:
            error_type = random.choice(["A", "B", "C", "D", "E", "F"])
        else:
            error_type = random.choice(["A", "B", "C", "D"])
        return self.inject_error(error_type)

    def inject_error_robust(self, error_type: str) -> InjectionResult:
        """
        Inject a specific type of error using robust methods.

        Robust methods have higher success rates by using intelligent
        selection strategies (slack-based, sensitivity analysis, etc.)

        Args:
            error_type: One of "A", "B", "C", "D", "E", "F"

        Returns:
            InjectionResult with details of the injection
        """
        type_map = {
            "A": self.inject_type_a_robust,
            "B": self.inject_type_b_robust,
            "C": self.inject_type_c_robust,
            "D": self.inject_type_d_robust,
            "E": self.inject_type_e_robust,
            "F": self.inject_type_f_robust,
        }

        if error_type not in type_map:
            raise ValueError(f"Unknown error type: {error_type}. Must be A, B, C, D, E, or F")

        return type_map[error_type]()

    def inject_random_error_robust(self, include_hard: bool = False) -> InjectionResult:
        """
        Inject a random error type using robust methods.

        Args:
            include_hard: If True, include Type E and F (harder problems)

        Returns:
            InjectionResult with details of the injection
        """
        if include_hard:
            error_type = random.choice(["A", "B", "C", "D", "E", "F"])
        else:
            error_type = random.choice(["A", "B", "C", "D"])
        return self.inject_error_robust(error_type)

    # =========================================================================
    # Helper Methods for IIS Extraction
    # =========================================================================

    def _compute_iis_info(self, model: gp.Model) -> tuple:
        """
        Compute IIS information from an infeasible model.

        Returns:
            Tuple of (iis_constraints, iis_bounds, iis_size)
        """
        model.computeIIS()
        iis_constraints = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
        iis_bounds = [v.VarName for v in model.getVars() if v.IISLB or v.IISUB]
        iis_size = len(iis_constraints) + len(iis_bounds)
        return iis_constraints, iis_bounds, iis_size

    def _get_original_objective(self) -> Optional[float]:
        """Get objective value from a solved model."""
        try:
            if self._model.Status == gp.GRB.OPTIMAL:
                return self._model.ObjVal
        except gp.GurobiError:
            pass
        return None

    # =========================================================================
    # Type A: Constraint Direction Flip
    # =========================================================================

    def inject_type_a(self) -> InjectionResult:
        """
        Type A: Flip constraint direction (≤ ↔ ≥).

        Selects a random inequality constraint and flips its sense.
        This often causes infeasibility when bounds become contradictory.

        Returns:
            InjectionResult with injection details

        Raises:
            ValueError: If no inequality constraints exist
        """
        # Find inequality constraints
        constrs = [
            c for c in self._model.getConstrs()
            if c.Sense != gp.GRB.EQUAL
        ]

        if not constrs:
            raise ValueError("No inequality constraints available for Type A injection")

        # Select random constraint
        target = random.choice(constrs)
        original_sense = target.Sense
        original_sense_str = "<=" if original_sense == gp.GRB.LESS_EQUAL else ">="

        # Flip the sense
        if original_sense == gp.GRB.LESS_EQUAL:
            target.Sense = gp.GRB.GREATER_EQUAL
            new_sense_str = ">="
            fix = f"Change {target.ConstrName} from >= back to <="
        else:
            target.Sense = gp.GRB.LESS_EQUAL
            new_sense_str = "<="
            fix = f"Change {target.ConstrName} from <= back to >="

        self._model.update()

        # Solve to check result
        state = self._solver.solve()

        result = InjectionResult(
            success=state.status in ["INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"],
            error_type=ErrorType.TYPE_A,
            target_name=target.ConstrName,
            original_value=original_sense_str,
            modified_value=new_sense_str,
            solver_status=state.status,
            ground_truth_fix=fix,
            metadata={"rhs": target.RHS}
        )

        self._injection_history.append(result)
        return result

    def inject_type_a_robust(self) -> InjectionResult:
        """
        Type A Robust: Slack-based constraint direction flip.

        Uses slack values to identify "tight" constraints where flipping
        is most likely to cause infeasibility.

        Strategy:
        1. Solve the original model to get slack values
        2. Sort constraints by |slack| ascending (tightest first)
        3. Iteratively try flipping until infeasibility is confirmed
        4. Verify IIS contains the flipped constraint

        Returns:
            InjectionResult with injection details

        Raises:
            ValueError: If no inequality constraints exist
        """
        # Get original objective before modification
        original_state = self._solver.solve()
        original_objective = original_state.objective if original_state.status == "OPTIMAL" else None

        # Find inequality constraints with slack info
        constrs = [
            c for c in self._model.getConstrs()
            if c.Sense != gp.GRB.EQUAL
        ]

        if not constrs:
            raise ValueError("No inequality constraints available for Type A injection")

        # Sort by slack (tightest first)
        slack_info = {}
        for c in constrs:
            try:
                slack_info[c.ConstrName] = abs(c.Slack)
            except gp.GurobiError:
                slack_info[c.ConstrName] = float('inf')

        candidates = sorted(constrs, key=lambda c: slack_info.get(c.ConstrName, float('inf')))

        # Try top 10 tightest constraints
        for target in candidates[:10]:
            # Create a copy to test
            model_copy = self._model.copy()
            test_constr = model_copy.getConstrByName(target.ConstrName)

            if test_constr is None:
                continue

            original_sense = test_constr.Sense
            original_sense_str = "<=" if original_sense == gp.GRB.LESS_EQUAL else ">="

            # Flip the sense
            if original_sense == gp.GRB.LESS_EQUAL:
                test_constr.Sense = gp.GRB.GREATER_EQUAL
                new_sense_str = ">="
            else:
                test_constr.Sense = gp.GRB.LESS_EQUAL
                new_sense_str = "<="

            model_copy.update()
            model_copy.setParam('OutputFlag', 0)
            model_copy.optimize()

            if model_copy.Status == gp.GRB.INFEASIBLE:
                # Verify target is in IIS
                try:
                    model_copy.computeIIS()
                    if test_constr.IISConstr:
                        # Success! Apply to actual model
                        actual_constr = self._model.getConstrByName(target.ConstrName)
                        if actual_constr.Sense == gp.GRB.LESS_EQUAL:
                            actual_constr.Sense = gp.GRB.GREATER_EQUAL
                            fix = f"Change {target.ConstrName} from >= back to <="
                        else:
                            actual_constr.Sense = gp.GRB.LESS_EQUAL
                            fix = f"Change {target.ConstrName} from <= back to >="

                        self._model.update()

                        # Get IIS info
                        iis_constraints, iis_bounds, iis_size = self._compute_iis_info(model_copy)
                        difficulty = Difficulty.from_iis_size(iis_size)

                        result = InjectionResult(
                            success=True,
                            error_type=ErrorType.TYPE_A,
                            target_name=target.ConstrName,
                            original_value=original_sense_str,
                            modified_value=new_sense_str,
                            solver_status="INFEASIBLE",
                            ground_truth_fix=fix,
                            metadata={"rhs": target.RHS, "slack": slack_info.get(target.ConstrName, 0)},
                            difficulty=difficulty,
                            iis_size=iis_size,
                            iis_constraints=iis_constraints,
                            iis_bounds=iis_bounds,
                            original_objective=original_objective
                        )

                        self._injection_history.append(result)
                        return result
                except gp.GurobiError:
                    continue

        # No successful injection found
        result = InjectionResult(
            success=False,
            error_type=ErrorType.TYPE_A,
            target_name="",
            original_value="",
            modified_value="",
            solver_status="FAILED",
            ground_truth_fix="",
            metadata={"reason": "No tight constraint found that causes infeasibility"},
        )
        self._injection_history.append(result)
        return result

    # =========================================================================
    # Type B: Variable Type Modification
    # =========================================================================

    def inject_type_b(self) -> InjectionResult:
        """
        Type B: Change variable type (INTEGER/BINARY ↔ CONTINUOUS).

        Selects a random integer/binary variable and changes it to continuous,
        or vice versa. This can cause subtle issues with integrality constraints.

        Returns:
            InjectionResult with injection details

        Raises:
            ValueError: If no integer/binary variables exist
        """
        # Find integer or binary variables
        int_vars = [
            v for v in self._model.getVars()
            if v.VType in [gp.GRB.BINARY, gp.GRB.INTEGER]
        ]

        if not int_vars:
            # Try to find continuous variables to make integer
            cont_vars = [
                v for v in self._model.getVars()
                if v.VType == gp.GRB.CONTINUOUS
            ]
            if not cont_vars:
                raise ValueError("No variables available for Type B injection")

            # Change continuous to integer
            target = random.choice(cont_vars)
            original_type = "C"
            target.VType = gp.GRB.INTEGER
            new_type = "I"
            fix = f"Change {target.VarName} from INTEGER back to CONTINUOUS"
        else:
            # Change integer/binary to continuous
            target = random.choice(int_vars)
            original_type = "B" if target.VType == gp.GRB.BINARY else "I"
            target.VType = gp.GRB.CONTINUOUS
            new_type = "C"
            fix = f"Change {target.VarName} from CONTINUOUS back to {original_type}"

        self._model.update()

        # Solve to check result
        state = self._solver.solve()

        result = InjectionResult(
            success=state.status in ["INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "SUBOPTIMAL"],
            error_type=ErrorType.TYPE_B,
            target_name=target.VarName,
            original_value=original_type,
            modified_value=new_type,
            solver_status=state.status,
            ground_truth_fix=fix,
            metadata={"lb": target.LB, "ub": target.UB}
        )

        self._injection_history.append(result)
        return result

    def inject_type_b_robust(self) -> InjectionResult:
        """
        Type B Robust: Variable type change with forcing constraint.

        Changes INTEGER variables to BINARY and adds a forcing constraint
        that guarantees infeasibility (e.g., x >= 2 when x is binary).

        Strategy:
        1. Find INTEGER variables with UB > 1
        2. Change to BINARY (0-1)
        3. Add forcing constraint: x >= 2 (impossible for binary)

        Returns:
            InjectionResult with injection details

        Raises:
            ValueError: If no suitable variables exist
        """
        # Get original objective before modification
        original_state = self._solver.solve()
        original_objective = original_state.objective if original_state.status == "OPTIMAL" else None

        # Find integer variables with UB > 1 (can be forced to conflict)
        int_vars = [
            v for v in self._model.getVars()
            if v.VType == gp.GRB.INTEGER and v.UB > 1
        ]

        if not int_vars:
            # Fallback: continuous variables with non-trivial bounds
            cont_vars = [
                v for v in self._model.getVars()
                if v.VType == gp.GRB.CONTINUOUS and v.UB > 1
            ]
            if cont_vars:
                int_vars = cont_vars

        if not int_vars:
            result = InjectionResult(
                success=False,
                error_type=ErrorType.TYPE_B,
                target_name="",
                original_value="",
                modified_value="",
                solver_status="FAILED",
                ground_truth_fix="",
                metadata={"reason": "No suitable variables with UB > 1"},
            )
            self._injection_history.append(result)
            return result

        # Try each candidate
        for target in int_vars:
            # Test on a copy first
            model_copy = self._model.copy()
            test_var = model_copy.getVarByName(target.VarName)

            if test_var is None:
                continue

            original_vtype = test_var.VType
            original_vtype_str = "I" if original_vtype == gp.GRB.INTEGER else "C"
            original_ub = test_var.UB

            # Change to binary
            test_var.VType = gp.GRB.BINARY
            test_var.UB = 1
            test_var.LB = 0

            # Add forcing constraint: x >= 2 (impossible for binary)
            forcing_value = 2
            forcing_constr_name = f"_force_{target.VarName}"
            model_copy.addConstr(test_var >= forcing_value, name=forcing_constr_name)
            model_copy.update()

            model_copy.setParam('OutputFlag', 0)
            model_copy.optimize()

            if model_copy.Status == gp.GRB.INFEASIBLE:
                # Success! Apply to actual model
                actual_var = self._model.getVarByName(target.VarName)
                actual_var.VType = gp.GRB.BINARY
                actual_var.UB = 1
                actual_var.LB = 0
                forcing_constr = self._model.addConstr(actual_var >= forcing_value, name=forcing_constr_name)
                self._model.update()

                # Get IIS info
                iis_constraints, iis_bounds, iis_size = self._compute_iis_info(model_copy)
                difficulty = Difficulty.from_iis_size(iis_size)

                fix = f"Change {target.VarName} back to INTEGER with UB={original_ub}, remove {forcing_constr_name}"

                result = InjectionResult(
                    success=True,
                    error_type=ErrorType.TYPE_B,
                    target_name=target.VarName,
                    original_value=original_vtype_str,
                    modified_value="B",
                    solver_status="INFEASIBLE",
                    ground_truth_fix=fix,
                    metadata={
                        "original_vtype": original_vtype_str,
                        "original_ub": original_ub,
                        "forcing_constraint": forcing_constr_name,
                        "forcing_value": forcing_value
                    },
                    difficulty=difficulty,
                    iis_size=iis_size,
                    iis_constraints=iis_constraints,
                    iis_bounds=iis_bounds,
                    original_objective=original_objective
                )

                self._injection_history.append(result)
                return result

        # No successful injection found
        result = InjectionResult(
            success=False,
            error_type=ErrorType.TYPE_B,
            target_name="",
            original_value="",
            modified_value="",
            solver_status="FAILED",
            ground_truth_fix="",
            metadata={"reason": "No variable found that causes infeasibility when changed to binary"},
        )
        self._injection_history.append(result)
        return result

    # =========================================================================
    # Type C: Expression Term Removal
    # =========================================================================

    def inject_type_c(self) -> InjectionResult:
        """
        Type C: Remove a term from a constraint expression.

        This modifies the constraint by changing the coefficient of a
        random variable to zero, effectively removing it from the expression.

        Returns:
            InjectionResult with injection details

        Raises:
            ValueError: If no suitable constraints exist
        """
        constrs = list(self._model.getConstrs())
        if not constrs:
            raise ValueError("No constraints available for Type C injection")

        # Try to find a constraint with multiple terms
        random.shuffle(constrs)

        for target in constrs:
            row = self._model.getRow(target)
            if row.size() > 1:
                # Select a random term to remove
                term_idx = random.randint(0, row.size() - 1)
                var = row.getVar(term_idx)
                original_coeff = row.getCoeff(term_idx)

                # Set coefficient to zero
                self._model.chgCoeff(target, var, 0.0)
                self._model.update()

                # Solve to check result
                state = self._solver.solve()

                fix = f"Restore coefficient of {var.VarName} in {target.ConstrName} to {original_coeff}"

                result = InjectionResult(
                    success=state.status in ["INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"],
                    error_type=ErrorType.TYPE_C,
                    target_name=target.ConstrName,
                    original_value=f"{var.VarName}: {original_coeff}",
                    modified_value=f"{var.VarName}: 0.0",
                    solver_status=state.status,
                    ground_truth_fix=fix,
                    metadata={
                        "variable": var.VarName,
                        "original_coeff": original_coeff,
                        "constraint": target.ConstrName
                    }
                )

                self._injection_history.append(result)
                return result

        raise ValueError("No constraints with multiple terms found for Type C injection")

    def inject_type_c_robust(self) -> InjectionResult:
        """
        Type C Robust: Strategic coefficient modification.

        Uses multiple strategies to cause infeasibility through coefficient changes:
        1. For >= constraints: remove positive terms (makes LHS smaller, can violate >= rhs)
        2. For <= constraints: flip coefficient signs (makes LHS larger, can violate <= rhs)
        3. Scale up coefficients dramatically to cause constraint violations

        Returns:
            InjectionResult with injection details

        Raises:
            ValueError: If no suitable constraints exist
        """
        # Get original objective before modification
        original_state = self._solver.solve()
        original_objective = original_state.objective if original_state.status == "OPTIMAL" else None

        if original_state.status != "OPTIMAL":
            result = InjectionResult(
                success=False,
                error_type=ErrorType.TYPE_C,
                target_name="",
                original_value="",
                modified_value="",
                solver_status="FAILED",
                ground_truth_fix="",
                metadata={"reason": "Original model not optimal, cannot compute duals"},
            )
            self._injection_history.append(result)
            return result

        # Collect all constraints with their properties
        constr_candidates = []
        for c in self._model.getConstrs():
            row = self._model.getRow(c)
            if row.size() >= 2:
                try:
                    dual = abs(c.Pi)
                except gp.GurobiError:
                    dual = 0
                constr_candidates.append((c, c.Sense, dual))

        # Sort by dual value (higher priority first)
        constr_candidates.sort(key=lambda x: x[2], reverse=True)

        # Strategy 1: For >= constraints, remove large positive terms
        for constr, sense, dual in constr_candidates:
            if sense != gp.GRB.GREATER_EQUAL:
                continue

            row = self._model.getRow(constr)
            # Get terms sorted by coefficient (largest positive first)
            terms = [(row.getVar(i), row.getCoeff(i)) for i in range(row.size())]
            terms.sort(key=lambda x: x[1], reverse=True)

            for var, coeff in terms:
                if coeff <= 0:
                    continue  # Only positive terms for >= constraints

                # Test removal
                model_copy = self._model.copy()
                test_constr = model_copy.getConstrByName(constr.ConstrName)
                test_var = model_copy.getVarByName(var.VarName)

                if test_constr is None or test_var is None:
                    continue

                model_copy.chgCoeff(test_constr, test_var, 0.0)
                model_copy.update()
                model_copy.setParam('OutputFlag', 0)
                model_copy.optimize()

                if model_copy.Status == gp.GRB.INFEASIBLE:
                    return self._finalize_type_c_injection(
                        constr, var, coeff, 0.0, model_copy, original_objective,
                        "removal"
                    )

        # Strategy 2: For <= constraints, flip positive coefficient signs
        for constr, sense, dual in constr_candidates:
            if sense != gp.GRB.LESS_EQUAL:
                continue

            row = self._model.getRow(constr)
            terms = [(row.getVar(i), row.getCoeff(i)) for i in range(row.size())]
            terms.sort(key=lambda x: abs(x[1]), reverse=True)

            for var, coeff in terms:
                if coeff <= 0:
                    continue  # Flip positive to negative

                # Test sign flip
                model_copy = self._model.copy()
                test_constr = model_copy.getConstrByName(constr.ConstrName)
                test_var = model_copy.getVarByName(var.VarName)

                if test_constr is None or test_var is None:
                    continue

                new_coeff = -coeff  # Flip sign
                model_copy.chgCoeff(test_constr, test_var, new_coeff)
                model_copy.update()
                model_copy.setParam('OutputFlag', 0)
                model_copy.optimize()

                if model_copy.Status == gp.GRB.INFEASIBLE:
                    return self._finalize_type_c_injection(
                        constr, var, coeff, new_coeff, model_copy, original_objective,
                        "sign_flip"
                    )

        # Strategy 3: Scale up coefficients dramatically
        for constr, sense, dual in constr_candidates[:5]:
            row = self._model.getRow(constr)
            terms = [(row.getVar(i), row.getCoeff(i)) for i in range(row.size())]

            for var, coeff in terms:
                if abs(coeff) < 0.1:
                    continue

                # Scale up by 10x
                model_copy = self._model.copy()
                test_constr = model_copy.getConstrByName(constr.ConstrName)
                test_var = model_copy.getVarByName(var.VarName)

                if test_constr is None or test_var is None:
                    continue

                new_coeff = coeff * 10
                model_copy.chgCoeff(test_constr, test_var, new_coeff)
                model_copy.update()
                model_copy.setParam('OutputFlag', 0)
                model_copy.optimize()

                if model_copy.Status == gp.GRB.INFEASIBLE:
                    return self._finalize_type_c_injection(
                        constr, var, coeff, new_coeff, model_copy, original_objective,
                        "scale_up"
                    )

        # Strategy 4: Guaranteed fallback - create tight constraints then modify coefficient
        # Pick any variable with non-zero optimal value
        vars_with_value = []
        for v in self._model.getVars():
            try:
                if abs(v.X) > 0.1:  # Variable has significant value
                    vars_with_value.append((v, v.X))
            except gp.GurobiError:
                pass

        if vars_with_value:
            target_var, opt_value = vars_with_value[0]

            # Create tight upper bound: var <= opt_value + small_margin
            upper_name = f"_saboteur_upper_{target_var.VarName}"
            upper_constr = self._model.addConstr(
                target_var <= opt_value + 0.5,
                name=upper_name
            )

            # Create lower bound with coefficient: coeff * var >= coeff * (opt_value - margin)
            # Initial: 1.0 * var >= opt_value - 0.5 (feasible since var = opt_value)
            original_coeff_val = 1.0
            lower_rhs = opt_value - 0.5
            lower_name = f"_saboteur_lower_{target_var.VarName}"
            lower_constr = self._model.addConstr(
                original_coeff_val * target_var >= lower_rhs,
                name=lower_name
            )
            self._model.update()

            # Verify model still optimal
            self._model.setParam('OutputFlag', 0)
            self._model.optimize()

            if self._model.Status != gp.GRB.OPTIMAL:
                # Clean up and fail
                self._model.remove(upper_constr)
                self._model.remove(lower_constr)
                self._model.update()
            else:
                # Now modify coefficient: 0.1 * var >= (opt_value - 0.5)
                # This requires var >= 10 * (opt_value - 0.5)
                # But upper bound is var <= opt_value + 0.5
                # If opt_value = 5, need var >= 45 but var <= 5.5 -> CONFLICT!
                new_coeff_val = 0.1
                self._model.chgCoeff(lower_constr, target_var, new_coeff_val)
                self._model.update()

                self._model.optimize()

                if self._model.Status == gp.GRB.INFEASIBLE:
                    # Get IIS info
                    iis_constraints, iis_bounds, iis_size = self._compute_iis_info(self._model)
                    difficulty = Difficulty.from_iis_size(iis_size)

                    fix = f"Change coefficient of {target_var.VarName} in {lower_name} from {new_coeff_val} back to {original_coeff_val}"

                    result = InjectionResult(
                        success=True,
                        error_type=ErrorType.TYPE_C,
                        target_name=f"{lower_name}:{target_var.VarName}",
                        original_value=f"{target_var.VarName}: {original_coeff_val}",
                        modified_value=f"{target_var.VarName}: {new_coeff_val}",
                        solver_status="INFEASIBLE",
                        ground_truth_fix=fix,
                        metadata={
                            "constraint": lower_name,
                            "variable": target_var.VarName,
                            "original_coeff": original_coeff_val,
                            "new_coeff": new_coeff_val,
                            "modification_type": "guaranteed_fallback",
                            "optimal_var_value": opt_value,
                            "upper_constraint": upper_name
                        },
                        difficulty=difficulty,
                        iis_size=iis_size,
                        iis_constraints=iis_constraints,
                        iis_bounds=iis_bounds,
                        original_objective=original_objective
                    )

                    self._injection_history.append(result)
                    return result
                else:
                    # Clean up if didn't cause infeasibility
                    self._model.remove(upper_constr)
                    self._model.remove(lower_constr)
                    self._model.update()

        # No successful injection found
        result = InjectionResult(
            success=False,
            error_type=ErrorType.TYPE_C,
            target_name="",
            original_value="",
            modified_value="",
            solver_status="FAILED",
            ground_truth_fix="",
            metadata={"reason": "No coefficient modification found that causes infeasibility"},
        )
        self._injection_history.append(result)
        return result

    def _finalize_type_c_injection(
        self,
        constr,
        var,
        original_coeff: float,
        new_coeff: float,
        model_copy: gp.Model,
        original_objective: Optional[float],
        modification_type: str
    ) -> InjectionResult:
        """Helper method to finalize Type C injection after successful test."""
        # Apply to actual model
        actual_constr = self._model.getConstrByName(constr.ConstrName)
        actual_var = self._model.getVarByName(var.VarName)
        self._model.chgCoeff(actual_constr, actual_var, new_coeff)
        self._model.update()

        # Get IIS info
        iis_constraints, iis_bounds, iis_size = self._compute_iis_info(model_copy)
        difficulty = Difficulty.from_iis_size(iis_size)

        if modification_type == "removal":
            fix = f"Restore coefficient of {var.VarName} in {constr.ConstrName} to {original_coeff}"
            modified_value = "0.0"
        elif modification_type == "sign_flip":
            fix = f"Change coefficient of {var.VarName} in {constr.ConstrName} from {new_coeff} back to {original_coeff}"
            modified_value = str(new_coeff)
        else:  # scale_up
            fix = f"Change coefficient of {var.VarName} in {constr.ConstrName} from {new_coeff} back to {original_coeff}"
            modified_value = str(new_coeff)

        result = InjectionResult(
            success=True,
            error_type=ErrorType.TYPE_C,
            target_name=f"{constr.ConstrName}:{var.VarName}",
            original_value=f"{var.VarName}: {original_coeff}",
            modified_value=f"{var.VarName}: {modified_value}",
            solver_status="INFEASIBLE",
            ground_truth_fix=fix,
            metadata={
                "constraint": constr.ConstrName,
                "variable": var.VarName,
                "original_coeff": original_coeff,
                "new_coeff": new_coeff,
                "modification_type": modification_type
            },
            difficulty=difficulty,
            iis_size=iis_size,
            iis_constraints=iis_constraints,
            iis_bounds=iis_bounds,
            original_objective=original_objective
        )

        self._injection_history.append(result)
        return result

    # =========================================================================
    # Type D: Contradicting Constraint Addition
    # =========================================================================

    def inject_type_d(self) -> InjectionResult:
        """
        Type D: Add a contradicting constraint.

        Creates a new constraint that directly contradicts an existing one,
        guaranteeing infeasibility.

        Returns:
            InjectionResult with injection details

        Raises:
            ValueError: If no suitable constraints exist
        """
        constrs = list(self._model.getConstrs())
        if not constrs:
            raise ValueError("No constraints available for Type D injection")

        # Select a random constraint to contradict
        target = random.choice(constrs)
        row = self._model.getRow(target)

        # Create contradicting constraint
        new_constr_name = f"_saboteur_conflict_{target.ConstrName}"

        # Strategy: Create opposite constraint with tighter bound
        if target.Sense == gp.GRB.LESS_EQUAL:
            # Original: expr <= rhs
            # Add: expr >= rhs + delta (contradicts for bounded expr)
            delta = abs(target.RHS) + 10 if target.RHS != 0 else 10
            new_rhs = target.RHS + delta
            new_constr = self._model.addConstr(
                row >= new_rhs,
                name=new_constr_name
            )
            original_desc = f"{target.ConstrName}: expr <= {target.RHS}"
            new_desc = f"expr >= {new_rhs}"
        elif target.Sense == gp.GRB.GREATER_EQUAL:
            # Original: expr >= rhs
            # Add: expr <= rhs - delta (contradicts for bounded expr)
            delta = abs(target.RHS) + 10 if target.RHS != 0 else 10
            new_rhs = target.RHS - delta
            new_constr = self._model.addConstr(
                row <= new_rhs,
                name=new_constr_name
            )
            original_desc = f"{target.ConstrName}: expr >= {target.RHS}"
            new_desc = f"expr <= {new_rhs}"
        else:
            # EQUAL constraint - add contradicting inequality
            delta = abs(target.RHS) + 10 if target.RHS != 0 else 10
            new_rhs = target.RHS + delta
            new_constr = self._model.addConstr(
                row >= new_rhs,
                name=new_constr_name
            )
            original_desc = f"{target.ConstrName}: expr = {target.RHS}"
            new_desc = f"expr >= {new_rhs}"

        self._model.update()

        # Solve to check result
        state = self._solver.solve()

        fix = f"Remove contradicting constraint {new_constr_name}"

        result = InjectionResult(
            success=state.status in ["INFEASIBLE", "INF_OR_UNBD"],
            error_type=ErrorType.TYPE_D,
            target_name=new_constr_name,
            original_value=original_desc,
            modified_value=new_desc,
            solver_status=state.status,
            ground_truth_fix=fix,
            metadata={
                "original_constraint": target.ConstrName,
                "new_constraint": new_constr_name,
                "original_rhs": target.RHS,
                "new_rhs": new_rhs
            }
        )

        self._injection_history.append(result)
        return result

    def inject_type_d_robust(self, target_iis_size: int = 2) -> InjectionResult:
        """
        Type D Robust: Contradicting constraint with IIS size control.

        Creates conflicts with controlled IIS size:
        - target_iis_size <= 2: Simple direct conflict (Easy)
        - target_iis_size > 2: Chain conflict through auxiliary variables (Medium/Hard)

        Args:
            target_iis_size: Target IIS size (default 2 for simple conflict)

        Returns:
            InjectionResult with injection details

        Raises:
            ValueError: If no suitable constraints exist
        """
        # Get original objective before modification
        original_state = self._solver.solve()
        original_objective = original_state.objective if original_state.status == "OPTIMAL" else None

        constrs = list(self._model.getConstrs())
        if not constrs:
            raise ValueError("No constraints available for Type D injection")

        if target_iis_size <= 2:
            # Simple direct conflict (Easy difficulty)
            return self._inject_simple_conflict(original_objective)
        else:
            # Chain conflict for larger IIS (Medium/Hard difficulty)
            return self._inject_chain_conflict(target_iis_size, original_objective)

    def _inject_simple_conflict(self, original_objective: Optional[float]) -> InjectionResult:
        """
        Create a simple direct conflict (IIS size 1-2).
        """
        constrs = list(self._model.getConstrs())
        target = random.choice(constrs)
        row = self._model.getRow(target)

        new_constr_name = f"_saboteur_conflict_{target.ConstrName}"

        # Create opposite constraint with tighter bound
        if target.Sense == gp.GRB.LESS_EQUAL:
            delta = abs(target.RHS) + 10 if target.RHS != 0 else 10
            new_rhs = target.RHS + delta
            new_constr = self._model.addConstr(row >= new_rhs, name=new_constr_name)
            original_desc = f"{target.ConstrName}: expr <= {target.RHS}"
            new_desc = f"expr >= {new_rhs}"
        elif target.Sense == gp.GRB.GREATER_EQUAL:
            delta = abs(target.RHS) + 10 if target.RHS != 0 else 10
            new_rhs = target.RHS - delta
            new_constr = self._model.addConstr(row <= new_rhs, name=new_constr_name)
            original_desc = f"{target.ConstrName}: expr >= {target.RHS}"
            new_desc = f"expr <= {new_rhs}"
        else:
            delta = abs(target.RHS) + 10 if target.RHS != 0 else 10
            new_rhs = target.RHS + delta
            new_constr = self._model.addConstr(row >= new_rhs, name=new_constr_name)
            original_desc = f"{target.ConstrName}: expr = {target.RHS}"
            new_desc = f"expr >= {new_rhs}"

        self._model.update()
        state = self._solver.solve()

        # Get IIS info
        if state.status in ["INFEASIBLE", "INF_OR_UNBD"]:
            iis_constraints, iis_bounds, iis_size = self._compute_iis_info(self._model)
        else:
            iis_constraints, iis_bounds, iis_size = [], [], 0

        difficulty = Difficulty.from_iis_size(iis_size)
        fix = f"Remove contradicting constraint {new_constr_name}"

        result = InjectionResult(
            success=state.status in ["INFEASIBLE", "INF_OR_UNBD"],
            error_type=ErrorType.TYPE_D,
            target_name=new_constr_name,
            original_value=original_desc,
            modified_value=new_desc,
            solver_status=state.status,
            ground_truth_fix=fix,
            metadata={
                "original_constraint": target.ConstrName,
                "new_constraint": new_constr_name,
                "conflict_type": "simple"
            },
            difficulty=difficulty,
            iis_size=iis_size,
            iis_constraints=iis_constraints,
            iis_bounds=iis_bounds,
            original_objective=original_objective
        )

        self._injection_history.append(result)
        return result

    def _inject_chain_conflict(self, target_iis_size: int, original_objective: Optional[float]) -> InjectionResult:
        """
        Create a chain conflict for larger IIS size.

        Creates a cycle of constraints through auxiliary variables:
        y >= x + 1, z >= y + 1, ..., x >= last_var + 1 (creates cycle)
        """
        chain_length = min(target_iis_size, 8)  # Cap at 8 to avoid complexity
        chain_vars = []
        chain_constrs = []

        # Create auxiliary variables
        for i in range(chain_length):
            var = self._model.addVar(lb=0, ub=100, name=f"_chain_var_{i}")
            chain_vars.append(var)

        self._model.update()

        # Create chain constraints: var[i] >= var[i-1] + 1
        for i in range(1, chain_length):
            constr = self._model.addConstr(
                chain_vars[i] >= chain_vars[i-1] + 10,
                name=f"_chain_constr_{i}"
            )
            chain_constrs.append(constr)

        # Close the cycle: var[0] >= var[last] + 1 (creates infeasibility)
        cycle_constr = self._model.addConstr(
            chain_vars[0] >= chain_vars[-1] + 10,
            name="_chain_cycle"
        )
        chain_constrs.append(cycle_constr)

        self._model.update()
        state = self._solver.solve()

        # Get IIS info
        if state.status in ["INFEASIBLE", "INF_OR_UNBD"]:
            iis_constraints, iis_bounds, iis_size = self._compute_iis_info(self._model)
        else:
            iis_constraints, iis_bounds, iis_size = [], [], 0

        difficulty = Difficulty.from_iis_size(iis_size)

        # Build fix description
        fix_parts = [f"Remove chain constraints: {', '.join(c.ConstrName for c in chain_constrs)}"]
        fix = "; ".join(fix_parts)

        result = InjectionResult(
            success=state.status in ["INFEASIBLE", "INF_OR_UNBD"],
            error_type=ErrorType.TYPE_D,
            target_name="_chain_cycle",
            original_value="No chain",
            modified_value=f"Chain of {chain_length} constraints",
            solver_status=state.status,
            ground_truth_fix=fix,
            metadata={
                "chain_length": chain_length,
                "chain_vars": [v.VarName for v in chain_vars],
                "chain_constrs": [c.ConstrName for c in chain_constrs],
                "conflict_type": "chain"
            },
            difficulty=difficulty,
            iis_size=iis_size,
            iis_constraints=iis_constraints,
            iis_bounds=iis_bounds,
            original_objective=original_objective
        )

        self._injection_history.append(result)
        return result

    # =========================================================================
    # Type E: Multi-Constraint Conflict (requires 2+ fixes)
    # =========================================================================

    def inject_type_e(self) -> InjectionResult:
        """
        Type E: Create multi-constraint conflict requiring 2+ fixes.

        This creates a situation where fixing just one constraint is not enough.
        Two or more constraints must be modified together to restore feasibility.

        Strategy:
        1. Create two tight constraints that share variables
        2. Both constraints need relaxation - fixing one alone fails

        Returns:
            InjectionResult with injection details
        """
        return self.inject_type_e_robust()

    def inject_type_e_robust(self) -> InjectionResult:
        """
        Type E Robust: Multi-constraint conflict with guaranteed 2+ fixes needed.

        Creates interlocked constraints:
        - Constraint E1: x + y <= small_value
        - Constraint E2: x + y >= large_value
        - Both must be fixed (can't satisfy both)

        Returns:
            InjectionResult with injection details
        """
        # Get original objective before modification
        original_state = self._solver.solve()
        original_objective = original_state.objective if original_state.status == "OPTIMAL" else None

        if original_state.status != "OPTIMAL":
            result = InjectionResult(
                success=False,
                error_type=ErrorType.TYPE_E,
                target_name="",
                original_value="",
                modified_value="",
                solver_status="FAILED",
                ground_truth_fix="",
                metadata={"reason": "Original model not optimal"},
            )
            self._injection_history.append(result)
            return result

        # Find two variables with non-zero optimal values
        vars_with_value = []
        for v in self._model.getVars():
            try:
                if abs(v.X) > 0.01:
                    vars_with_value.append((v, v.X))
            except gp.GurobiError:
                pass

        if len(vars_with_value) < 2:
            result = InjectionResult(
                success=False,
                error_type=ErrorType.TYPE_E,
                target_name="",
                original_value="",
                modified_value="",
                solver_status="FAILED",
                ground_truth_fix="",
                metadata={"reason": "Not enough variables with non-zero values"},
            )
            self._injection_history.append(result)
            return result

        # Select two variables
        var1, val1 = vars_with_value[0]
        var2, val2 = vars_with_value[1]
        sum_val = val1 + val2

        # Create conflicting constraints:
        # E1: var1 + var2 <= sum_val - margin (tight upper)
        # E2: var1 + var2 >= sum_val + margin (tight lower)
        # These two together are infeasible!
        margin = max(1.0, abs(sum_val) * 0.1)

        constr_e1_name = f"_multi_constr_upper_{var1.VarName}_{var2.VarName}"
        constr_e2_name = f"_multi_constr_lower_{var1.VarName}_{var2.VarName}"

        # Upper constraint: sum <= small_value
        upper_rhs = sum_val - margin
        constr_e1 = self._model.addConstr(
            var1 + var2 <= upper_rhs,
            name=constr_e1_name
        )

        # Lower constraint: sum >= large_value
        lower_rhs = sum_val + margin
        constr_e2 = self._model.addConstr(
            var1 + var2 >= lower_rhs,
            name=constr_e2_name
        )

        self._model.update()
        state = self._solver.solve()

        if state.status not in ["INFEASIBLE", "INF_OR_UNBD"]:
            # Remove constraints and fail
            self._model.remove(constr_e1)
            self._model.remove(constr_e2)
            self._model.update()

            result = InjectionResult(
                success=False,
                error_type=ErrorType.TYPE_E,
                target_name="",
                original_value="",
                modified_value="",
                solver_status=state.status,
                ground_truth_fix="",
                metadata={"reason": "Constraints did not cause infeasibility"},
            )
            self._injection_history.append(result)
            return result

        # Get IIS info
        iis_constraints, iis_bounds, iis_size = self._compute_iis_info(self._model)
        difficulty = Difficulty.HARD  # Type E is always hard

        # Ground truth fix requires modifying BOTH constraints
        fix = (
            f"Remove BOTH constraints: {constr_e1_name} AND {constr_e2_name}. "
            f"Fixing only one will not restore feasibility."
        )

        result = InjectionResult(
            success=True,
            error_type=ErrorType.TYPE_E,
            target_name=f"{constr_e1_name},{constr_e2_name}",
            original_value="No conflict",
            modified_value=f"sum <= {upper_rhs} AND sum >= {lower_rhs}",
            solver_status="INFEASIBLE",
            ground_truth_fix=fix,
            metadata={
                "constraint_1": constr_e1_name,
                "constraint_2": constr_e2_name,
                "variables": [var1.VarName, var2.VarName],
                "original_sum": sum_val,
                "upper_rhs": upper_rhs,
                "lower_rhs": lower_rhs,
                "num_fixes_required": 2,
            },
            difficulty=difficulty,
            iis_size=iis_size,
            iis_constraints=iis_constraints,
            iis_bounds=iis_bounds,
            original_objective=original_objective
        )

        self._injection_history.append(result)
        return result

    # =========================================================================
    # Type F: Hidden Dependency (root cause not directly in IIS)
    # =========================================================================

    def inject_type_f(self) -> InjectionResult:
        """
        Type F: Create hidden dependency where root cause is not in IIS.

        The IIS shows symptoms but not the root cause. The model needs to
        reason about WHY the IIS constraints conflict.

        Strategy:
        1. Create an auxiliary variable with a derived bound
        2. Add constraint using auxiliary var that causes conflict
        3. The IIS shows the derived constraint, not the root cause bound

        Returns:
            InjectionResult with injection details
        """
        return self.inject_type_f_robust()

    def inject_type_f_robust(self) -> InjectionResult:
        """
        Type F Robust: Hidden dependency with indirect causation.

        Creates structure:
        - aux_var defined by: aux = expr(original_vars)
        - root_cause: aux >= large_value (the actual error)
        - symptom: Uses aux in way that conflicts with original constraints
        - IIS shows symptom constraints, not the root cause

        Returns:
            InjectionResult with injection details
        """
        # Get original objective before modification
        original_state = self._solver.solve()
        original_objective = original_state.objective if original_state.status == "OPTIMAL" else None

        if original_state.status != "OPTIMAL":
            result = InjectionResult(
                success=False,
                error_type=ErrorType.TYPE_F,
                target_name="",
                original_value="",
                modified_value="",
                solver_status="FAILED",
                ground_truth_fix="",
                metadata={"reason": "Original model not optimal"},
            )
            self._injection_history.append(result)
            return result

        # Find a variable with bounded optimal value
        target_var = None
        target_val = None
        for v in self._model.getVars():
            try:
                if 0.1 < abs(v.X) < 100:  # Reasonable range
                    target_var = v
                    target_val = v.X
                    break
            except gp.GurobiError:
                pass

        if target_var is None:
            result = InjectionResult(
                success=False,
                error_type=ErrorType.TYPE_F,
                target_name="",
                original_value="",
                modified_value="",
                solver_status="FAILED",
                ground_truth_fix="",
                metadata={"reason": "No suitable variable found"},
            )
            self._injection_history.append(result)
            return result

        # Create auxiliary variable
        aux_var_name = f"_hidden_aux_{target_var.VarName}"
        aux_var = self._model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=aux_var_name)
        self._model.update()

        # Define aux_var = target_var (linking constraint - this shows in IIS)
        linking_name = f"_hidden_link_{target_var.VarName}"
        linking_constr = self._model.addConstr(
            aux_var == target_var,
            name=linking_name
        )

        # ROOT CAUSE: Set impossible bound on aux_var
        # If target_val = 5, require aux >= 100 (impossible since aux == target)
        impossible_bound = abs(target_val) * 20 + 50
        root_cause_name = f"_hidden_root_{target_var.VarName}"
        root_cause_constr = self._model.addConstr(
            aux_var >= impossible_bound,
            name=root_cause_name
        )

        self._model.update()
        state = self._solver.solve()

        if state.status not in ["INFEASIBLE", "INF_OR_UNBD"]:
            # Clean up and fail
            self._model.remove(linking_constr)
            self._model.remove(root_cause_constr)
            self._model.remove(aux_var)
            self._model.update()

            result = InjectionResult(
                success=False,
                error_type=ErrorType.TYPE_F,
                target_name="",
                original_value="",
                modified_value="",
                solver_status=state.status,
                ground_truth_fix="",
                metadata={"reason": "Hidden dependency did not cause infeasibility"},
            )
            self._injection_history.append(result)
            return result

        # Get IIS info
        iis_constraints, iis_bounds, iis_size = self._compute_iis_info(self._model)
        difficulty = Difficulty.HARD  # Type F is always hard

        # The IIS will likely show the linking constraint and original bounds
        # but the ROOT CAUSE is the impossible bound on aux_var
        fix = (
            f"The root cause is constraint '{root_cause_name}' which requires "
            f"{aux_var_name} >= {impossible_bound}, but {aux_var_name} is linked to "
            f"{target_var.VarName} (optimal value ~{target_val:.2f}). "
            f"Remove or relax '{root_cause_name}'."
        )

        # Check if root cause is visible in IIS
        root_cause_in_iis = root_cause_name in iis_constraints

        result = InjectionResult(
            success=True,
            error_type=ErrorType.TYPE_F,
            target_name=root_cause_name,
            original_value=f"{target_var.VarName} = {target_val:.2f}",
            modified_value=f"{aux_var_name} >= {impossible_bound}",
            solver_status="INFEASIBLE",
            ground_truth_fix=fix,
            metadata={
                "root_cause_constraint": root_cause_name,
                "linking_constraint": linking_name,
                "auxiliary_variable": aux_var_name,
                "original_variable": target_var.VarName,
                "original_value": target_val,
                "impossible_bound": impossible_bound,
                "root_cause_in_iis": root_cause_in_iis,
                "reasoning_required": not root_cause_in_iis,
            },
            difficulty=difficulty,
            iis_size=iis_size,
            iis_constraints=iis_constraints,
            iis_bounds=iis_bounds,
            original_objective=original_objective
        )

        self._injection_history.append(result)
        return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def validate_injection(self) -> bool:
        """
        Validate that the model is now infeasible/unbounded.

        Returns:
            True if model is no longer optimal/feasible
        """
        state = self._solver.solve()
        return state.status in ["INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"]

    def get_injection_history(self) -> List[InjectionResult]:
        """Get the history of all injections performed."""
        return self._injection_history.copy()

    def get_last_injection(self) -> Optional[InjectionResult]:
        """Get the most recent injection result."""
        if self._injection_history:
            return self._injection_history[-1]
        return None

    @property
    def solver(self) -> GurobiSolver:
        """Access the underlying solver."""
        return self._solver

    @property
    def model(self) -> gp.Model:
        """Access the underlying Gurobi model."""
        return self._model
