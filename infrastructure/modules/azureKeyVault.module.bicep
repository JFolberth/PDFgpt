@description('Name for the Azure Key Vault')
param keyVaultName string
@description('Sku for the Azure Key Vault')
@allowed(['standard', 'premium'])
param sku string = 'standard'
@description('Location for resource.')
param location string
@description('What language was used to deploy this resource')
param language string

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01'={
name: toLower('kv-${keyVaultName}')
location: location
tags: {
    language: language
}
properties: {
    sku: {
        name: sku
        family: 'A'
    }
    tenantId: subscription().tenantId
    enabledForDeployment: true
    enabledForTemplateDeployment: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    enableRbacAuthorization: true
}
}

output keyVaultNameOutput string = keyVault.name
