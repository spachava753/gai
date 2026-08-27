package gai

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"strings"
	"testing"
)

// TestExportedAPIHasDocumentation prevents undocumented public API additions.
// It parses production files so exported fields and interface methods are
// covered alongside top-level declarations.
func TestExportedAPIHasDocumentation(t *testing.T) {
	files, err := parser.ParseDir(token.NewFileSet(), ".", func(info fs.FileInfo) bool {
		return !strings.HasSuffix(info.Name(), "_test.go")
	}, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	pkg := files["gai"]
	for filename, file := range pkg.Files {
		for _, declaration := range file.Decls {
			switch declaration := declaration.(type) {
			case *ast.FuncDecl:
				if ast.IsExported(declaration.Name.Name) && exportedReceiver(declaration) && declaration.Doc == nil {
					t.Errorf("%s: exported function or method %s has no documentation", filename, declaration.Name.Name)
				}
			case *ast.GenDecl:
				checkExportedSpecs(t, filename, declaration)
			}
		}
	}
}

func exportedReceiver(declaration *ast.FuncDecl) bool {
	if declaration.Recv == nil {
		return true
	}

	receiver := declaration.Recv.List[0].Type
	if pointer, ok := receiver.(*ast.StarExpr); ok {
		receiver = pointer.X
	}
	identifier, ok := receiver.(*ast.Ident)
	return ok && ast.IsExported(identifier.Name)
}

func checkExportedSpecs(t *testing.T, filename string, declaration *ast.GenDecl) {
	t.Helper()

	for _, rawSpec := range declaration.Specs {
		switch spec := rawSpec.(type) {
		case *ast.TypeSpec:
			if !ast.IsExported(spec.Name.Name) {
				continue
			}
			if spec.Doc == nil && declaration.Doc == nil {
				t.Errorf("%s: exported type %s has no documentation", filename, spec.Name.Name)
			}
			checkExportedFields(t, filename, spec)
		case *ast.ValueSpec:
			for _, name := range spec.Names {
				if !ast.IsExported(name.Name) {
					continue
				}
				if spec.Doc == nil && spec.Comment == nil && !(len(declaration.Specs) == 1 && declaration.Doc != nil) {
					t.Errorf("%s: exported value %s has no documentation", filename, name.Name)
				}
			}
		}
	}
}

func checkExportedFields(t *testing.T, filename string, spec *ast.TypeSpec) {
	t.Helper()

	var fields *ast.FieldList
	switch value := spec.Type.(type) {
	case *ast.StructType:
		fields = value.Fields
	case *ast.InterfaceType:
		fields = value.Methods
	default:
		return
	}

	for _, field := range fields.List {
		if field.Doc != nil || field.Comment != nil {
			continue
		}
		for _, name := range field.Names {
			if ast.IsExported(name.Name) {
				t.Errorf("%s: exported field or interface method %s.%s has no documentation", filename, spec.Name.Name, name.Name)
			}
		}
	}
}
