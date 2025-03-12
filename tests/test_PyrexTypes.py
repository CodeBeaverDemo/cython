import pytest
import copy
import re
from Cython.Compiler.PyrexTypes import BaseType, PyrexType, CTypedefType
from Cython.Compiler.PyrexTypes import c_const_type, public_decl, FusedType
from Cython.Compiler.PyrexTypes import CVoidType, InvisibleVoidType, CIntType, CNumericType, PyObjectType
from Cython.Compiler.PyrexTypes import CArrayType
from Cython.Compiler.PyrexTypes import CPyTSSTType
from Cython.Compiler.PyrexTypes import CFloatType, CComplexType
from Cython.Compiler.PyrexTypes import CConstOrVolatileType, CPyUnicodeIntType, CPySSizeTType, CSSizeTType, CSizeTType, CPtrdiffTType

# Define a dummy type for testing purposes by subclassing BaseType.
class DummyType(BaseType):
    is_cv_qualified = 0
    has_attributes = False
    scope = None
    is_complex = False
    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        return "dummy" + entity_code
    def resolve(self):
        return self

    def same_as(self, other, **kwds):
        return self is other

    def assignable_from(self, src_type):
        return self.same_as(src_type)

    def deduce_template_params(self, actual):
        return {}

    def py_type_name(self):
        return None
    def specialize(self, values):
        return self
    def error_condition(self, result_code):
        return None

# Define a dummy Pyrex type to test behavior dependent on declaration_code.
class DummyPyrexType(PyrexType):
    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        return "unsigned int" + entity_code

class TestPyrexTypes:
    # Helper classes for the new tests
    class DummyFuncState:
        def __init__(self):
            self.needs_refnanny = False

    class DummyCode:
        def __init__(self):
            self.lines = []
            self.funcstate = TestPyrexTypes.DummyFuncState()
        def putln(self, line):
            self.lines.append(line)
    class DummyGlobalState:
        def __init__(self):
            self.used_utilities = []
        def use_utility_code(self, code):
            self.used_utilities.append(code)

    # A dummy environment that records calls to use_utility_code
    class DummyEnv:
        def __init__(self):
            self.used_utilities = []
        def use_utility_code(self, code):
            self.used_utilities.append(code)

    # A dummy numeric type subclassing CIntType to simulate integer behavior
    class DummyCInt(CIntType):
        def __init__(self):
            # Call super with rank 1 (for example) and signed True.
            super().__init__(rank=1, signed=1)
        def sign_and_name(self):
            return "dummy_int"
        def empty_declaration_code(self, pyrex=False):
            return "dummy_int"
        def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
            return "dummy_int" + entity_code

    def test_empty_declaration_code(self):
        """Test that empty_declaration_code calls declaration_code and caches its result."""
        dt = DummyType()
        decl1 = dt.empty_declaration_code()
        decl2 = dt.empty_declaration_code()
        assert decl1 == "dummy"
        assert decl1 is decl2  # Check the caching works

    def test_specialization_name(self):
        """Test that specialization_name replaces spaces and substrings correctly."""
        dt = DummyPyrexType()
        spec_name = dt.specialization_name()
        # For "unsigned int", declaration_code returns "unsigned int" which becomes "unsigned__int"
        assert "unsigned_int" in spec_name

    def test_cast_code(self):
        """Test that cast_code returns an expression that casts the given code correctly."""
        dt = DummyType()
        cast = dt.cast_code("x")
        # Should follow the format "((<empty_decl>)x)" where empty_decl from DummyType is "dummy"
        assert cast == "((dummy)x)"

    def test_deepcopy(self):
        """Test that __deepcopy__ returns self (since types are not meant to be copied)."""
        dt = DummyType()
        dt_copy = copy.deepcopy(dt)
        assert dt is dt_copy

    def test_ctypedef_type_declaration(self):
        """Test that a CTypedefType delegates its declaration_code correctly."""
        base = DummyType()
        ctypedef = CTypedefType("mytype", base, "mytype_cname")
        # When in pyrex mode or for display, the typedef_name is used.
        decl_pyrex = ctypedef.declaration_code("", for_display=True, pyrex=True)
        assert decl_pyrex == "mytype"
        # For non-pyrex mode, the typedef_cname is used.
        decl_c = ctypedef.declaration_code("", for_display=False, pyrex=False)
        assert "mytype_cname" in decl_c

    def test_lt_operator(self):
        """Test that the __lt__ method returns a boolean value for BaseType instances."""
        a = DummyType()
        b = DummyType()
        res = a < b
        assert isinstance(res, bool)

    def test_get_fused_types(self):
        """Test the get_fused_types method in a dummy subclass containing a subtype."""
        # Create a dummy type with a 'sub' attribute.
        class DummyWithSub(BaseType):
            subtypes = ['sub']
            def __init__(self, sub):
                self.sub = sub
            def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
                return "subdummy"

        dt = DummyWithSub(DummyType())
        fused = dt.get_fused_types()
        # By default, get_fused_types should return None when there are no fused types.
        assert fused in (None, [])
        # Now override get_fused_types to simulate a fused type.
        def custom_get_fused_types(self, result=None, seen=None, subtypes=None, include_function_return_type=False):
            return ["fused"]
        DummyWithSub.get_fused_types = custom_get_fused_types
        fused = dt.get_fused_types()
        assert fused == ["fused"]

    def test_invalid_value_in_dummy(self):
        """Test that invalid_value either returns None or raises NotImplementedError in DummyType."""
        dt = DummyType()
        try:
            val = dt.invalid_value()
        except NotImplementedError:
            val = None
        assert val is None

    def test_py_type_name_method(self):
        """Test that py_type_name returns None in DummyType (since it is not implemented)."""
        dt = DummyType()
        assert dt.py_type_name() is None

    def test_check_for_null_code(self):
        """Test that check_for_null_code returns None by default."""
        dt = DummyType()
        assert dt.check_for_null_code("x") is None
    def test_base_declaration_code(self):
        """Test that base_declaration_code correctly concatenates the base and entity codes."""
        dt = DummyType()
        # When entity_code is not empty, expect "base entity" (with a single space)
        result = dt.base_declaration_code("base", "entity")
        assert result == "base entity"
        # When entity_code is empty, it should return just the base_code.
        result_empty = dt.base_declaration_code("base", "")
        assert result_empty == "base"

    def test_deduce_template_params(self):
        """Test that deduce_template_params returns an empty dictionary by default."""
        dt = DummyType()
        params = dt.deduce_template_params("any_actual")
        assert params == {}

    def test_can_coercion_methods(self):
        """Test that can_coerce_to_pyobject and can_coerce_from_pyobject return False for DummyType."""
        dt = DummyType()
        assert dt.can_coerce_to_pyobject(None) is False
        assert dt.can_coerce_from_pyobject(None) is False

    def test_same_as_and_assignable_from(self):
        """Test that same_as returns True for self and assignable_from returns True when comparing to self."""
        dt = DummyType()
        # same_as should be True for an object compared to itself.
        assert dt.same_as(dt)
        # For assignable_from, since DummyType simply checks same_as, it should also return True.
        assert dt.assignable_from(dt)

    def test_resolve(self):
        """Test that resolve returns self for a dummy type."""
        dt = DummyType()
        resolved = dt.resolve()
        assert resolved is dt

    def test_ctypedef_getattr(self):
        """Test that CTypedefType delegates attribute lookups to its base type."""
        base = DummyType()
        base.custom = "custom_value"
        ctypedef = CTypedefType("alias", base, "alias_cname")
        # __getattr__ should return the attribute from the base type.
        assert ctypedef.custom == "custom_value"

    def test_convert_to_pystring_not_implemented(self):
        """Test that convert_to_pystring raises NotImplementedError when not overridden."""
        dt = DummyType()
        with pytest.raises(NotImplementedError):
            dt.convert_to_pystring("x", code=type("DummyCode", (), {"error_goto_if": lambda self, cond, pos: cond}))

    def test_specialize_fused_no_fused(self):
        """Test that specialize_fused returns self when the environment does not provide fused_to_specific."""
        dt = DummyType()
        # Create a dummy environment without the fused_to_specific attribute.
        env = type("DummyEnv", (), {"fused_to_specific": None})()
        specialized = dt.specialize_fused(env)
        # Since env.fused_to_specific is not set, specialize_fused should return self.
        assert specialized is dt
    def test_specialize_dummy_pyrex(self):
        """Test that specialize on a DummyPyrexType returns self."""
        dpt = DummyPyrexType()
        # Since PyrexType.specialize returns self, passing an empty dict must return self.
        specialized = dpt.specialize({})
        assert specialized is dpt

    def test_fused_type_specialize(self):
        """Test that FusedType.specialize raises CannotSpecialize when no substitution is provided."""
        fused = FusedType([DummyType()], name="dummy_fused")
        with pytest.raises(Exception) as excinfo:
            _ = fused.specialize({})
        from Cython.Compiler.PyrexTypes import CannotSpecialize
        assert isinstance(excinfo.value, CannotSpecialize)

    def test_cconst_type_declaration(self):
        """Test that a const type declaration prepends 'const ' to the base declaration."""
        dt = DummyType()
        const_type = c_const_type(dt)
        # DummyType.declaration_code returns "dummy" + entity_code so with empty entity_code we expect "const dummy"
        decl = const_type.declaration_code("")
        # If we are in non-pyrex mode, the "const " should get inserted in front.
        assert "const" in decl and "dummy" in decl

    def test_public_decl_function(self):
        """Test that public_decl correctly processes dll_linkage."""
        # When a dll_linkage is provided, the base_code should be wrapped.
        result = public_decl("base, more", "DL_EXPORT")
        # It should replace the comma with the marker
        assert "DL_EXPORT" in result and "__PYX_COMMA" in result

    def test_deepcopy_ctypedef(self):
        """Test that deepcopy on a CTypedefType returns the same instance."""
        base = DummyType()
        ctypedef = CTypedefType("alias", base, "alias_cname")
        copy_ctypedef = copy.deepcopy(ctypedef)
        assert ctypedef is copy_ctypedef

    def test_same_and_assignable_from_different_instance(self):
        """Test that two distinct DummyType instances are not considered the same."""
        dt1 = DummyType()
        dt2 = DummyType()
        # same_as in DummyType checks identity so should return False for different objects.
        assert dt1.same_as(dt2) is False
        # assignable_from also calls same_as so should return False.
        assert dt1.assignable_from(dt2) is False
    def test_cvoid_declaration(self):

        """Test that CVoidType.declaration_code returns an appropriate 'void' declaration."""
        cv = CVoidType()
        # In for_display mode with pyrex=True, we expect the string "void" 
        decl_display = cv.declaration_code("", for_display=True, pyrex=True)
        assert "void" in decl_display
        # In non‐pyrex mode the public declaration is used
        decl_c = cv.declaration_code("", for_display=False, pyrex=False)
        assert "void" in decl_c

    def test_invisible_void_declaration(self):
# End of tests.
        """Test that InvisibleVoidType.declaration_code returns [void] in display mode and an empty declaration otherwise."""
        iv = InvisibleVoidType()
        decl_display = iv.declaration_code("", for_display=True, pyrex=True)
        assert "[void]" in decl_display
        decl_c = iv.declaration_code("", for_display=False, pyrex=False)
        # public_decl returns an empty base string when dll_linkage is not provided
        assert decl_c.strip() == ""

    def test_dummy_numeric_specialization(self):
        """Create a dummy numeric type (subclassing CNumericType) and check that its specialization name contains the custom type name."""
        class DummyNumeric(CNumericType):
            def __init__(self):
                super().__init__(rank=1, signed=1)
            def sign_and_name(self):
                return "dummy_int"
            def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
                return "dummy_int" + entity_code

        dn = DummyNumeric()
        spec_name = dn.specialization_name()
        assert "dummy_int" in spec_name

    def test_pyobject_global_init(self):
        """Test that PyObjectType.global_init_code sets a variable to Py_None via a dummy code helper."""
        dummy_entry = type("DummyEntry", (), {})()
        dummy_entry.name = "dummy_var"
        class DummyCode:
            def __init__(self):
                self.lines = []
            def put_init_var_to_py_none(self, entry, nanny):
                self.lines.append(f"Init {entry.name} to Py_None")

        py_obj = PyObjectType()
        dummy_code = DummyCode()
        py_obj.global_init_code(dummy_entry, dummy_code)
        assert any("Py_None" in line for line in dummy_code.lines)

    def test_fused_type_repr(self):
        """Test that the __repr__ method of FusedType returns a string including its name."""
        ft = FusedType([DummyType()], name="dummy_fused")
        rep = repr(ft)
        assert "dummy_fused" in rep

    def test_specialize_fused_with_env(self):
        """Test that specialize_fused returns a specialization via specialize when fused_to_specific is not None."""
        dt = DummyType()
        # Create a dummy environment with a non-None fused_to_specific attribute.
        class DummyEnv:
            fused_to_specific = {"dummy": "value"}
        env = DummyEnv()
        specialized = dt.specialize_fused(env)
        # In DummyType, specialize is not overridden, so dt.specialize({}) simply returns dt.
        assert specialized is dt

    def test_error_condition(self):
        """Test that the default error_condition returns None for DummyType."""
        dt = DummyType()
        ec = dt.error_condition("x")
        assert ec is None
    def test_pyobject_as_pyobject(self):
        """Test that PyObjectType.as_pyobject returns the input when type is complete."""
        py_obj = PyObjectType()
        # In PyObjectType, is_complete() returns 1 and is_extension_type is 0, so as_pyobject returns cname unmodified.
        assert py_obj.as_pyobject("test_var") == "test_var"

    def test_cconst_or_volatile_declaration(self):
        """Test that a const type (created via c_const_type) prepends 'const ' to the base declaration."""
        dt = DummyType()
        const_dt = c_const_type(dt)
        decl = const_dt.declaration_code("", for_display=True, dll_linkage=None, pyrex=True)
        # Since DummyType.declaration_code returns "dummy"+entity_code, we expect 'const ' to be prepended.
        assert decl.startswith("const ") and "dummy" in decl

    def test_generate_decref_in_pyobject(self):
        """Test that PyObjectType.generate_decref outputs a decref call."""
        py_obj = PyObjectType()
        dummy_code = self.DummyCode()
        # call generate_decref with nanny=True and have_gil=True; make sure the line contains "Py_DECREF"
        py_obj.generate_decref(dummy_code, "var", nanny=True, have_gil=True)
        assert any(("Py_DECREF(" in line or "__Pyx_DECREF(" in line) for line in dummy_code.lines)
        assert any("__Pyx_DECREF(" in line for line in dummy_code.lines)

    def test_cint_type_overflow_check(self):
        """Test that DummyCInt.overflow_check_binop returns a string that includes binop and specialization name."""
        dummy_int = self.DummyCInt()
        env = self.DummyEnv()
        # Call with binop 'lshift'
        op_str = dummy_int.overflow_check_binop("lshift", env, const_rhs=False)
        # Since sign_and_name returns "dummy_int", check that the result contains these substrings.
        assert "lshift" in op_str and "dummy_int" in op_str

    def test_cint_type_convert_to_pystring(self):
        """Test that convert_to_pystring produces a conversion function call pattern for CIntType."""
        dummy_int = self.DummyCInt()
        # Ensure that the PyUnicode conversion utility is not yet set
        dummy_int.to_pyunicode_utility = None
        dummy_global = self.DummyGlobalState()
        # Create a dummy code object with globalstate attribute for capturing utility code usage.
        class DummyCodeWithGlobal:
            def __init__(self):
                self.lines = []
                self.globalstate = dummy_global
            def putln(self, line):
                self.lines.append(line)
        dummy_code = DummyCodeWithGlobal()
        # Call convert_to_pystring with a valid format spec e.g. "05d"
        result = dummy_int.convert_to_pystring("42", dummy_code, format_spec="05d")
        # The returned string should be a function call that starts with the conversion function cname.
        assert result.startswith("__Pyx_PyUnicode_From_") and "42" in result

    def test_int_like_parse_format_valid(self):
        """Test that _parse_format correctly parses a valid format specifier."""
        fmt, width, pad = CIntType._parse_format("05d")
        assert fmt == "d"
        assert width == 5
        assert pad == "0"

    def test_int_like_parse_format_invalid(self):
        """Test that _parse_format returns (None, 0, ' ') for an invalid format specifier."""
        fmt, width, pad = CIntType._parse_format("invalid")
        assert fmt is None
        assert width == 0
        assert pad == " "
    def test_error_condition_numeric(self):
        """Test that error_condition in a numeric type returns a proper check string."""
        # Use DummyCInt from the existing DummyCInt inner class which is a subclass of CIntType.
        dummy_int = self.DummyCInt()
        # For CIntType, error_condition builds a string using sign_and_name and exception_value.
        cond = dummy_int.error_condition("result")
        # Since DummyCInt.sign_and_name returns "dummy_int" and exception_value is -1,
        # we expect the condition to mention "result == (dummy_int)-1" and "PyErr_Occurred()".
        assert "result == (dummy_int)-1" in cond
        assert "PyErr_Occurred()" in cond

    class DummyCCharType(DummyType):
        is_string = True
        def __init__(self):
            super().__init__()
            # Force the is_string flag to be True regardless of CPointerBaseType checks.
            self.is_string = True
        def same_as(self, other, **kwds):
            # Simulate that if the other type's declaration is "char", then they match.
            if hasattr(other, "declaration_code"):
                return other.declaration_code("", for_display=True, pyrex=True) == "char"
            return False
        def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
            return "char" + entity_code
        def literal_code(self, value):
            import Cython.Compiler.StringEncoding as StringEncoding
            return '"%s"' % StringEncoding.escape_byte_string(value)

    def test_dummy_cchar_literal(self):
        """Test that DummyCCharType.literal_code returns the correctly escaped literal string."""
        dcc = self.DummyCCharType()
        lit = dcc.literal_code(b"hello")
        assert lit == '"hello"'
    def test_carray_literal_nonstring(self):
        """Test literal_code for a CArrayType whose base type does not trigger the string branch."""
        # Use DummyType (which does not simulate a char type) as the base.
        arr = CArrayType(DummyType(), 5)
        # In this case, literal_code should fall back to str(value)
        lit = arr.literal_code("123")
        assert lit == "123"

    def test_cpytsst_declaration(self):
        """Test that CPyTSSTType.declaration_code returns a proper 'Py_tss_t' declaration."""
        tss = CPyTSSTType()
        decl_disp = tss.declaration_code("", for_display=True, pyrex=True)
        decl_c = tss.declaration_code("", for_display=False, pyrex=False)
        assert "Py_tss_t" in decl_disp
        assert "Py_tss_t" in decl_c

    def test_ccomplex_methods(self):
        """Test basic methods of CComplexType, including declaration_code, real_code and imag_code."""
        # Create a CFloatType to use as the real type.
        float_type = CFloatType(rank=1, math_h_modifier='')
        complex_type = CComplexType(float_type)
        # For display mode, declaration_code should include "complex" and use the real_type's declaration.
        decl = complex_type.declaration_code(" var", for_display=True, pyrex=True)
        # Check that the declaration string is non-empty and contains "complex" or the real type string.
        assert "complex" in decl or "float" in decl.lower()
        # Test real_code and imag_code produce expected function-call strings.
        rcode = complex_type.real_code("z")
        icode = complex_type.imag_code("z")
        expected_real = "__Pyx_CREAL%s(z)" % complex_type.implementation_suffix
        expected_imag = "__Pyx_CIMAG%s(z)" % complex_type.implementation_suffix
        assert rcode == expected_real
    def test_cconst_or_volatile_methods(self):
        """Test CConstOrVolatileType cv_string and delegation of same_as."""
        base = DummyType()
        cvt = CConstOrVolatileType(base, is_const=True, is_volatile=True)
        assert cvt.cv_string() == "volatile const "
        # Test delegation: same_as should delegate to base type's same_as
        other = DummyType()
        assert cvt.same_as(other) == base.same_as(other)

    def test_cfloat_type_declaration(self):
        """Test that CFloatType.declaration_code returns a declaration containing its sign_and_name."""
        cf = CFloatType(rank=1, math_h_modifier='')
        decl = cf.declaration_code("", for_display=True, pyrex=True)
        assert cf.sign_and_name() in decl

    def test_cpyunicodeint_type(self):
        """Test CPyUnicodeIntType declaration contains 'Py_UNICODE'."""
        from Cython.Compiler.PyrexTypes import CPyUnicodeIntType
        cui = CPyUnicodeIntType(1)
        decl = cui.declaration_code("", for_display=True, pyrex=True)
        assert "Py_UNICODE" in decl

    def test_cptrdiff_and_size_types(self):
        """Test declarations for CPySSizeTType, CSSizeTType, CSizeTType, and CPtrdiffTType."""
        from Cython.Compiler.PyrexTypes import CPySSizeTType, CSSizeTType, CSizeTType, CPtrdiffTType
        ssize = CPySSizeTType(1)
        assert "Py_ssize_t" in ssize.declaration_code("", for_display=True, pyrex=True)
        cssize = CSSizeTType(1)
        assert "Py_ssize_t" in cssize.declaration_code("", for_display=True, pyrex=True)
        csize = CSizeTType(1)
        assert "size_t" in csize.declaration_code("", for_display=True, pyrex=True)
        cptrdiff = CPtrdiffTType(1)
        assert "ptrdiff_t" in cptrdiff.declaration_code("", for_display=True, pyrex=True)

    def test_pythran_expr_equality(self):
        """Test equality and inequality for PythranExpr instances."""
        from Cython.Compiler.PyrexTypes import PythranExpr
        pe1 = PythranExpr("foo")
        pe2 = PythranExpr("foo")
        pe3 = PythranExpr("bar")
        assert pe1 == pe2
        assert pe1 != pe3

    def test_pyextension_type_type_test(self):
        """Test PyExtensionType.type_test_code produces a type test string."""
        from Cython.Compiler.PyrexTypes import PyExtensionType
        pe = PyExtensionType("TestType", True, None)
        pe.module_name = "__builtin__"
        pe.objstruct_cname = "TestTypeObj"
        pe.typeptr_cname = "TestTypePtr"
        class DummyScope:
            class_name = "DummyScope"
            def name_in_module_state(self, name):
                return name
            def use_utility_code(self, code):
                pass
        pe.scope = DummyScope()
        s = pe.type_test_code(pe.scope, "arg", allow_none=True)
        assert "__Pyx_TypeTest" in s