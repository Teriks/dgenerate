# Generated from Config.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .Config import Config
else:
    from Config import Config

# This class defines a complete generic visitor for a parse tree produced by Config.

class ConfigVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by Config#config.
    def visitConfig(self, ctx:Config.ConfigContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#element.
    def visitElement(self, ctx:Config.ElementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#stmt.
    def visitStmt(self, ctx:Config.StmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#continuation.
    def visitContinuation(self, ctx:Config.ContinuationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#piece.
    def visitPiece(self, ctx:Config.PieceContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jinjaExpr.
    def visitJinjaExpr(self, ctx:Config.JinjaExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jinjaFor.
    def visitJinjaFor(self, ctx:Config.JinjaForContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#forElse.
    def visitForElse(self, ctx:Config.ForElseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jinjaIf.
    def visitJinjaIf(self, ctx:Config.JinjaIfContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#ifBody.
    def visitIfBody(self, ctx:Config.IfBodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#elifPart.
    def visitElifPart(self, ctx:Config.ElifPartContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#elsePart.
    def visitElsePart(self, ctx:Config.ElsePartContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jinjaSet.
    def visitJinjaSet(self, ctx:Config.JinjaSetContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jinjaFilter.
    def visitJinjaFilter(self, ctx:Config.JinjaFilterContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jinjaWith.
    def visitJinjaWith(self, ctx:Config.JinjaWithContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jAssign.
    def visitJAssign(self, ctx:Config.JAssignContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jTargets.
    def visitJTargets(self, ctx:Config.JTargetsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jExpr.
    def visitJExpr(self, ctx:Config.JExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jOr.
    def visitJOr(self, ctx:Config.JOrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jAnd.
    def visitJAnd(self, ctx:Config.JAndContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jNot.
    def visitJNot(self, ctx:Config.JNotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jComparison.
    def visitJComparison(self, ctx:Config.JComparisonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jCmpOp.
    def visitJCmpOp(self, ctx:Config.JCmpOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jConcat.
    def visitJConcat(self, ctx:Config.JConcatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jSum.
    def visitJSum(self, ctx:Config.JSumContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jTerm.
    def visitJTerm(self, ctx:Config.JTermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jFactor.
    def visitJFactor(self, ctx:Config.JFactorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jPower.
    def visitJPower(self, ctx:Config.JPowerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jPostfix.
    def visitJPostfix(self, ctx:Config.JPostfixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jSuffix.
    def visitJSuffix(self, ctx:Config.JSuffixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jPrimary.
    def visitJPrimary(self, ctx:Config.JPrimaryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jParenItems.
    def visitJParenItems(self, ctx:Config.JParenItemsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jList.
    def visitJList(self, ctx:Config.JListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jDict.
    def visitJDict(self, ctx:Config.JDictContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jDictPair.
    def visitJDictPair(self, ctx:Config.JDictPairContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jArgs.
    def visitJArgs(self, ctx:Config.JArgsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Config#jArg.
    def visitJArg(self, ctx:Config.JArgContext):
        return self.visitChildren(ctx)



del Config