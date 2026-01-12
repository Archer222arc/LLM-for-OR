"""
Four-phase validation pipeline for OR-Debug-Bench data.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/03_BENCH.md

Key Components:
    - ProblemValidator: Main validation class
    - ValidationResult: Result of validation
    - Four-phase validation: Original → Sabotaged → IIS → Fix

Example:
    >>> from src.data_generation import ProblemValidator
    >>> validator = ProblemValidator()
    >>> result = validator.validate_problem(original, sabotaged, injection_result)
    >>> print(f"Valid: {result.is_valid}")
"""

from dataclasses import dataclass
from typing import Optional, List, Any

import gurobipy as gp

from src.solvers import GurobiSolver
from .error_types import InjectionResult, ErrorType


@dataclass
class ValidationResult:
    """Result of validation pipeline."""

    is_valid: bool
    phase_results: dict
    error_message: Optional[str] = None

    @property
    def phase1_passed(self) -> bool:
        """Original model is optimal."""
        return self.phase_results.get("phase1", {}).get("passed", False)

    @property
    def phase2_passed(self) -> bool:
        """Sabotaged model is infeasible."""
        return self.phase_results.get("phase2", {}).get("passed", False)

    @property
    def phase3_passed(self) -> bool:
        """Target is in IIS."""
        return self.phase_results.get("phase3", {}).get("passed", False)

    @property
    def phase4_passed(self) -> bool:
        """Fix restores feasibility."""
        return self.phase_results.get("phase4", {}).get("passed", False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "phase_results": self.phase_results,
            "error_message": self.error_message,
        }


class ProblemValidator:
    """
    Four-phase validation pipeline for OR-Debug-Bench problems.

    Phase 1: Original model must be optimal
    Phase 2: Sabotaged model must be infeasible
    Phase 3: IIS must contain the injection target
    Phase 4: Applying the fix must restore feasibility
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize validator.

        Args:
            verbose: If True, print validation details
        """
        self.verbose = verbose

    def validate_problem(
        self,
        original_model: gp.Model,
        sabotaged_model: gp.Model,
        injection_result: InjectionResult,
    ) -> ValidationResult:
        """
        Run four-phase validation on a problem.

        Args:
            original_model: The original feasible model
            sabotaged_model: The model after error injection
            injection_result: Details of the error injection

        Returns:
            ValidationResult with pass/fail for each phase
        """
        phase_results = {}

        # Phase 1: Original model must be optimal
        phase1 = self._validate_phase1(original_model)
        phase_results["phase1"] = phase1
        if not phase1["passed"]:
            return ValidationResult(
                is_valid=False,
                phase_results=phase_results,
                error_message=f"Phase 1 failed: {phase1['message']}"
            )

        # Phase 2: Sabotaged model must be infeasible
        phase2 = self._validate_phase2(sabotaged_model)
        phase_results["phase2"] = phase2
        if not phase2["passed"]:
            return ValidationResult(
                is_valid=False,
                phase_results=phase_results,
                error_message=f"Phase 2 failed: {phase2['message']}"
            )

        # Phase 3: IIS must contain target
        phase3 = self._validate_phase3(sabotaged_model, injection_result)
        phase_results["phase3"] = phase3
        if not phase3["passed"]:
            return ValidationResult(
                is_valid=False,
                phase_results=phase_results,
                error_message=f"Phase 3 failed: {phase3['message']}"
            )

        # Phase 4: Fix must restore feasibility
        phase4 = self._validate_phase4(sabotaged_model, injection_result)
        phase_results["phase4"] = phase4
        if not phase4["passed"]:
            return ValidationResult(
                is_valid=False,
                phase_results=phase_results,
                error_message=f"Phase 4 failed: {phase4['message']}"
            )

        return ValidationResult(
            is_valid=True,
            phase_results=phase_results,
        )

    def _validate_phase1(self, model: gp.Model) -> dict:
        """
        Phase 1: Validate original model is optimal.

        Args:
            model: Original model

        Returns:
            Dict with 'passed' and 'message'
        """
        try:
            model_copy = model.copy()
            model_copy.setParam('OutputFlag', 0)
            model_copy.optimize()

            if model_copy.Status == gp.GRB.OPTIMAL:
                return {
                    "passed": True,
                    "message": "Original model is optimal",
                    "status": "OPTIMAL",
                    "objective": model_copy.ObjVal
                }
            else:
                status_map = {
                    gp.GRB.INFEASIBLE: "INFEASIBLE",
                    gp.GRB.UNBOUNDED: "UNBOUNDED",
                    gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
                }
                status_str = status_map.get(model_copy.Status, str(model_copy.Status))
                return {
                    "passed": False,
                    "message": f"Original model is not optimal: {status_str}",
                    "status": status_str
                }
        except gp.GurobiError as e:
            return {
                "passed": False,
                "message": f"Gurobi error: {str(e)}",
                "status": "ERROR"
            }

    def _validate_phase2(self, model: gp.Model) -> dict:
        """
        Phase 2: Validate sabotaged model is infeasible.

        Args:
            model: Sabotaged model

        Returns:
            Dict with 'passed' and 'message'
        """
        try:
            model_copy = model.copy()
            model_copy.setParam('OutputFlag', 0)
            model_copy.optimize()

            if model_copy.Status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
                return {
                    "passed": True,
                    "message": "Sabotaged model is infeasible",
                    "status": "INFEASIBLE"
                }
            else:
                status_map = {
                    gp.GRB.OPTIMAL: "OPTIMAL",
                    gp.GRB.UNBOUNDED: "UNBOUNDED",
                }
                status_str = status_map.get(model_copy.Status, str(model_copy.Status))
                return {
                    "passed": False,
                    "message": f"Sabotaged model is not infeasible: {status_str}",
                    "status": status_str
                }
        except gp.GurobiError as e:
            return {
                "passed": False,
                "message": f"Gurobi error: {str(e)}",
                "status": "ERROR"
            }

    def _validate_phase3(self, model: gp.Model, injection_result: InjectionResult) -> dict:
        """
        Phase 3: Validate IIS contains the injection target.

        Args:
            model: Sabotaged model
            injection_result: Details of the injection

        Returns:
            Dict with 'passed' and 'message'
        """
        try:
            model_copy = model.copy()
            model_copy.setParam('OutputFlag', 0)
            model_copy.optimize()

            if model_copy.Status not in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
                return {
                    "passed": False,
                    "message": "Cannot compute IIS: model not infeasible",
                    "iis_constraints": [],
                    "iis_bounds": []
                }

            model_copy.computeIIS()

            iis_constraints = [c.ConstrName for c in model_copy.getConstrs() if c.IISConstr]
            iis_bounds = [v.VarName for v in model_copy.getVars() if v.IISLB or v.IISUB]

            target = injection_result.target_name

            # Check if target is in IIS
            target_in_constraints = target in iis_constraints
            target_in_bounds = target in iis_bounds

            # For Type C, target format is "constr:var", check constraint part
            if injection_result.error_type == ErrorType.TYPE_C and ":" in target:
                constr_name = target.split(":")[0]
                target_in_constraints = constr_name in iis_constraints

            # For Type B with forcing constraint, check the forcing constraint
            if injection_result.error_type == ErrorType.TYPE_B:
                forcing_name = injection_result.metadata.get("forcing_constraint", "")
                target_in_constraints = target_in_constraints or (forcing_name in iis_constraints)
                target_in_bounds = target_in_bounds or (target in iis_bounds)

            if target_in_constraints or target_in_bounds:
                return {
                    "passed": True,
                    "message": f"Target '{target}' found in IIS",
                    "iis_constraints": iis_constraints,
                    "iis_bounds": iis_bounds,
                    "iis_size": len(iis_constraints) + len(iis_bounds)
                }
            else:
                return {
                    "passed": False,
                    "message": f"Target '{target}' not in IIS. IIS: {iis_constraints + iis_bounds}",
                    "iis_constraints": iis_constraints,
                    "iis_bounds": iis_bounds,
                    "iis_size": len(iis_constraints) + len(iis_bounds)
                }

        except gp.GurobiError as e:
            return {
                "passed": False,
                "message": f"Gurobi error computing IIS: {str(e)}",
                "iis_constraints": [],
                "iis_bounds": []
            }

    def _validate_phase4(self, model: gp.Model, injection_result: InjectionResult) -> dict:
        """
        Phase 4: Validate that applying the fix restores feasibility.

        Args:
            model: Sabotaged model
            injection_result: Details of the injection

        Returns:
            Dict with 'passed' and 'message'
        """
        try:
            model_copy = model.copy()
            model_copy.setParam('OutputFlag', 0)

            # Apply fix based on error type
            success = self._apply_fix(model_copy, injection_result)

            if not success:
                return {
                    "passed": False,
                    "message": "Failed to apply fix",
                    "status": "FIX_FAILED"
                }

            model_copy.optimize()

            if model_copy.Status == gp.GRB.OPTIMAL:
                return {
                    "passed": True,
                    "message": "Fix restored feasibility",
                    "status": "OPTIMAL",
                    "objective": model_copy.ObjVal
                }
            else:
                status_map = {
                    gp.GRB.INFEASIBLE: "INFEASIBLE",
                    gp.GRB.UNBOUNDED: "UNBOUNDED",
                    gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
                }
                status_str = status_map.get(model_copy.Status, str(model_copy.Status))
                return {
                    "passed": False,
                    "message": f"Fix did not restore feasibility: {status_str}",
                    "status": status_str
                }

        except gp.GurobiError as e:
            return {
                "passed": False,
                "message": f"Gurobi error applying fix: {str(e)}",
                "status": "ERROR"
            }

    def _apply_fix(self, model: gp.Model, injection_result: InjectionResult) -> bool:
        """
        Apply the ground truth fix to the model.

        Args:
            model: Model to fix (will be modified in place)
            injection_result: Details of the injection

        Returns:
            True if fix was applied successfully
        """
        error_type = injection_result.error_type

        try:
            if error_type == ErrorType.TYPE_A:
                # Flip constraint sense back
                constr = model.getConstrByName(injection_result.target_name)
                if constr is None:
                    return False

                # Flip back to original
                if constr.Sense == gp.GRB.LESS_EQUAL:
                    constr.Sense = gp.GRB.GREATER_EQUAL
                else:
                    constr.Sense = gp.GRB.LESS_EQUAL
                model.update()
                return True

            elif error_type == ErrorType.TYPE_B:
                # Restore variable type and remove forcing constraint
                var = model.getVarByName(injection_result.target_name)
                if var is None:
                    return False

                # Restore type
                original_vtype = injection_result.metadata.get("original_vtype", "I")
                if original_vtype == "I":
                    var.VType = gp.GRB.INTEGER
                else:
                    var.VType = gp.GRB.CONTINUOUS

                # Restore bounds
                original_ub = injection_result.metadata.get("original_ub", gp.GRB.INFINITY)
                var.UB = original_ub

                # Remove forcing constraint if present
                forcing_name = injection_result.metadata.get("forcing_constraint", "")
                if forcing_name:
                    forcing_constr = model.getConstrByName(forcing_name)
                    if forcing_constr:
                        model.remove(forcing_constr)

                model.update()
                return True

            elif error_type == ErrorType.TYPE_C:
                # Restore coefficient
                metadata = injection_result.metadata
                constr_name = metadata.get("constraint", "")
                var_name = metadata.get("variable", "")
                original_coeff = metadata.get("original_coeff", 0)

                constr = model.getConstrByName(constr_name)
                var = model.getVarByName(var_name)

                if constr is None or var is None:
                    return False

                model.chgCoeff(constr, var, original_coeff)
                model.update()
                return True

            elif error_type == ErrorType.TYPE_D:
                # Remove the added conflicting constraint(s)
                conflict_type = injection_result.metadata.get("conflict_type", "simple")

                if conflict_type == "chain":
                    # Remove all chain constraints
                    chain_constrs = injection_result.metadata.get("chain_constrs", [])
                    for constr_name in chain_constrs:
                        constr = model.getConstrByName(constr_name)
                        if constr:
                            model.remove(constr)

                    # Remove chain variables
                    chain_vars = injection_result.metadata.get("chain_vars", [])
                    for var_name in chain_vars:
                        var = model.getVarByName(var_name)
                        if var:
                            model.remove(var)
                else:
                    # Simple conflict - just remove the added constraint
                    constr = model.getConstrByName(injection_result.target_name)
                    if constr:
                        model.remove(constr)

                model.update()
                return True

            return False

        except gp.GurobiError:
            return False


def validate_dataset(problems: List[dict], verbose: bool = False) -> dict:
    """
    Validate a dataset of problems.

    Args:
        problems: List of problem dictionaries with 'original_model',
                  'sabotaged_model', and 'injection_result' keys
        verbose: If True, print validation details

    Returns:
        Summary statistics of validation
    """
    validator = ProblemValidator(verbose=verbose)

    total = len(problems)
    passed = 0
    failed_by_phase = {1: 0, 2: 0, 3: 0, 4: 0}

    for i, problem in enumerate(problems):
        result = validator.validate_problem(
            problem["original_model"],
            problem["sabotaged_model"],
            problem["injection_result"]
        )

        if result.is_valid:
            passed += 1
        else:
            # Determine which phase failed
            for phase in [1, 2, 3, 4]:
                if not result.phase_results.get(f"phase{phase}", {}).get("passed", True):
                    failed_by_phase[phase] += 1
                    break

        if verbose and (i + 1) % 100 == 0:
            print(f"Validated {i + 1}/{total} problems...")

    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total > 0 else 0,
        "failed_by_phase": failed_by_phase,
    }
